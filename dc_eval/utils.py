from collections import defaultdict, Counter
import math
import warnings
from typing import List

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from dc_eval.tasks.utils import evaluate


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        n_tasks=None,
        n_copies=1,
        prefix="",
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        # Ensure n_tasks does not exceed dataset length
        self.n_tasks = min(n_tasks if n_tasks is not None else len(dataset), len(dataset))
        self.n_copies = n_copies
        self.prefix = prefix

    def __iter__(self):
        prompts = []
        for sample in range(self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if not isinstance(prompt_contents, str):
                raise ValueError(f"Only string prompts are supported, got: {type(prompt_contents)}")
            prompt = self.prefix + prompt_contents
            prompts.append(prompt)

        self.tokenizer.padding_side = 'right'
        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=None,
        )

        n_copies = self.n_copies
        if n_copies == 1 and self.n_tasks % self.num_devices != 0:
            n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(n_copies):
                yield {
                    "ids": outputs.input_ids[sample],
                    "task_id": sample,
                    "input_len": outputs.attention_mask[sample].sum(),
                }


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    batch_size=20,
    prefix="",
    postprocess=True,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    gen_token_dict = defaultdict(list)
    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        with torch.no_grad():
            if task.stop_words:
                gen_kwargs["stopping_criteria"][0].start_length = batch["input_len"].max().item()
            input_ids = batch["ids"][:, :batch["input_len"].max().item()]
            first_dev = next(accelerator.unwrap_model(model).parameters()).device
            input_ids = input_ids.to(first_dev, non_blocking=True)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=first_dev)
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=batch_size,
                **gen_kwargs,
            )
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_tokens)

    code_gens_raw = [[] for _ in range(n_tasks)]
    code_gens_prc = [[] for _ in range(n_tasks)]
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            gen_code = tokenizer.decode(
                s, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            code_gens_raw[sample].append(gen_code[len(prefix) :])
            if postprocess:
                x = int(sample)
                code_gens_prc[sample].append(
                    task.postprocess_generation(gen_code[len(prefix) :], x)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens_prc[sample].append(gen_code[len(prefix) :])

    return code_gens_prc, code_gens_raw


def complete_code_custom(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    batch_size=20,
    n_samples=None,
    prefix="",
    postprocess=True,
    n_split_trials=10,
    **gen_kwargs,
):
    """dcneurosymbolic pipeline wrapper with consensus splitting (negation removed).

     Updated flow (no negation flags):
     1) For each sample: run the splitter n_split_trials times; select the most common operator,
         then among splits with that operator, select the most common children tuple.
     2) With the consensus (op, children) fixed, run LLM+prover for n_samples iterations per child.
     3) Aggregate per-child majority values via the selected operator.
     """
    from dc_eval.dc_helpers.splitting import run_split  # local import to avoid circulars
    from dc_eval.dc_helpers.parsing import parse_and_eval_generation
    from dc_eval.dc_helpers.three_valued_logic import combine_operator
    from dc_eval.dc_llm_splitter.dnc_llm_splitter.config import load_settings
    from dc_eval.dc_llm_splitter.dnc_llm_splitter.run_pipeline_enhanced import make_llm

    # Load splitter config + LLM once
    from pathlib import Path as _Path
    cfg_path = _Path(__file__).parent / "dc_llm_splitter" / "config.yaml"
    cfg = load_settings(str(cfg_path))
    # Reuse the already-loaded HF model/tokenizer to avoid a second CUDA load
    device = next(accelerator.unwrap_model(model).parameters()).device
    llm = make_llm(cfg, shared_model=accelerator.unwrap_model(model), shared_tokenizer=tokenizer, device=device)
    final_answers = [[] for _ in range(n_tasks)]
    final_raw = [[] for _ in range(n_tasks)]

    base_dataset = dataloader.dataset.dataset
    stop_words = task.stop_words or []
    error_token = getattr(task, 'ERROR_TOKEN', 'Error')
    max_len = gen_kwargs.get("max_length", dataloader.dataset.max_length)

    # Main evaluation loop per sample
    for sample_idx in tqdm(range(n_tasks), total=n_tasks):
        raw = []
        original_doc = base_dataset[sample_idx]
        conclusion_text = original_doc.get("conclusion", "")
        premises_texts = original_doc.get("premises", [])
        
        # 1) Build consensus split without passing to prover yet (no negation)
        results = []
        for n_idx in range(n_split_trials):
            try:
                tmp_op, tmp_children = run_split(conclusion_text, cfg, llm, strategy="rule")
                results.append((tmp_op, tuple(tmp_children)))
                raw.append(f"Split attempt {n_idx}: op={tmp_op}, children={tmp_children}")
            except Exception:
                print(f"Error: splitting attempt fails for sample {sample_idx}, trial {n_idx}")
                continue

        if not results:
            op, children = "NONE", [conclusion_text]
        else:
            op_counts = Counter(op for op, _ in results)
            chosen_op = op_counts.most_common(1)[0][0]
            combo_counts = Counter(children for op, children in results if op == chosen_op)
            chosen_children, _ = combo_counts.most_common(1)[0]
            op, children = chosen_op, list(chosen_children)
            
        final_raw[sample_idx].append(raw)
        raw = []
        
        # 2) With consensus decomposition, run LLM+prover n_samples times PER CHILD
        num_runs = n_samples if n_samples is not None else 1

        child_majorities = []
        for c_text in children:
                # Build child-specific prompt once
                doc_copy = dict(original_doc)
                doc_copy["conclusion"] = c_text

                prompt = task.get_prompt(doc_copy)
                full_prompt = prefix + prompt if prefix else prompt

                tokenizer.padding_side = 'right'
                enc = tokenizer(
                    [full_prompt],
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_len,
                    return_token_type_ids=None,
                )
                input_ids = enc.input_ids.to(device)
                attention_mask = enc.attention_mask.to(device)

                # Adjust stopping criteria start length if provided
                if stop_words and "stopping_criteria" in gen_kwargs:
                    try:
                        gen_kwargs["stopping_criteria"][0].start_length = attention_mask.sum(1).max().item()
                    except Exception:
                        pass

                per_child_vals = []
                for s_idx in range(num_runs):
                    with torch.no_grad():
                        out_tokens = accelerator.unwrap_model(model).generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_return_sequences=1,
                            **gen_kwargs,
                        )

                    if out_tokens.dim() == 3:
                        out_tokens = out_tokens.view(-1, out_tokens.size(-1))

                    decoded = tokenizer.batch_decode(
                        out_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    
                    val = parse_and_eval_generation(decoded[0], full_prompt, stop_words, error_token)
                    raw.append(f"Prover result {s_idx}: {val}, Child text: {c_text}, Prompt + Generation: {decoded[0]}")
                    per_child_vals.append(val)
                final_raw[sample_idx].append(raw)
                raw = []
                # 3) Pick the most common value across n_samples for this child
                per_child_vals = [v for v in per_child_vals if v != error_token]
                if len(per_child_vals) == 0: per_child_vals = ["Uncertain"]
                child_final = Counter(per_child_vals).most_common(1)[0][0]
                child_majorities.append(child_final)
        # 5) Aggregate the per-child majority values via the consensus operator once
        final_ans = combine_operator(op, child_majorities, error_token)
        # 6) Store one result per sample (post-consensus per-child majority)
        final_answers[sample_idx].append(final_ans)
        raw.append(f"Conclusion: '{conclusion_text}', Premises: {premises_texts}")
        raw.append(f"Split majority: op={op}, children={children}")
        raw.append(f"Child majorities: {child_majorities}")
        raw.append(f"Final answer after applying operator {op}: {final_ans}")
        final_raw[sample_idx].append(raw)

    return final_answers, final_raw

