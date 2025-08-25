"""Lightweight package initializer for dnc_llm_splitter (copy).

Avoid eager imports here to prevent pulling in heavy pipeline modules at
import time. Import modules/functions directly from their subpackages
where needed, e.g.:

from dnc_llm_splitter.run_pipeline_enhanced import run_pipeline_sentence
from dnc_llm_splitter.io_utils import read_conclusions

"""

__all__: list[str] = []
