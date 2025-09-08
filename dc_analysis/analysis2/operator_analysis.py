from collections import defaultdict
import json
import glob
import os
from json import JSONDecodeError

# Find the generations JSON file under /app without hardcoding substrings
def find_generations_file(root="/app", suffix="linc/outputs/Mistral-7B-v0.1_folio-dcneurosymbolic-1shot_generations_raw.json"):
    matches = glob.glob(os.path.join(root, "**", f"*{suffix}"), recursive=True)
    if not matches:
        return None
    # return the first match (adjust selection logic if needed)
    return matches[0]

data_path = find_generations_file()
if not data_path:
    raise FileNotFoundError("Unable to find any '*generations_raw.json' file under /app")

if os.path.getsize(data_path) == 0:
    raise ValueError(f"Found file {data_path} but it is empty")

with open(data_path, "r") as fh:
    try:
        # Read file content and fix JSON formatting issues
        content = fh.read()
        # Check for malformed prefix like 'nor[' at the beginning
        if content.startswith('nor['):
            content = content.replace('nor[', '[', 1)
        # Handle any other common JSON issues that might be present
        data = json.loads(content)
    except JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from {data_path}: {e}") from e

operator_groups = defaultdict(list)

print(f"Analyzing data from: {data_path}")
print(f"Found {len(data)} total items to process")

for idx, item in enumerate(data):
    if not isinstance(item, list) or not item:
        print(f"Skipping item {idx}: Not a list or empty")
        continue
    
    # Look for the last element which should contain our data
    last = item[-1]
    
    if not isinstance(last, list) or len(last) < 2:
        if idx < 10:  # Only show details for first few items to avoid console spam
            print(f"Skipping item {idx}: Last element not a list or too short")
        continue
    
    # Extract the split majority operator
    second = last[1]
    if isinstance(second, str) and second.startswith("Split majority: op="):
        if second != "Split majority: op=NONE":
            op = second.split("op=")[1].split(",")[0].strip()
            operator_groups[op].append(idx)

# Sort operators by frequency for better output
sorted_ops = sorted(operator_groups.items(), key=lambda x: len(x[1]), reverse=True)

print("\n===== RESULTS =====")
print(f"Found {len(operator_groups)} unique operators")

# Print results
total_items = len(data)

for op, indices in sorted_ops:
    count = len(indices)
    pct = (count / total_items) * 100 if total_items else 0.0
    print(f"{op}: {count} cases ({pct:.1f}% of {total_items})")
    # Print first few indices as examples
    print(f"Example indices: {indices[:5]}..." if len(indices) > 5 else indices)
    print()

# Save results to file for further analysis
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analysis2")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "operator_analysis_results.json")

results = {
    "source_file": data_path,
    "total_items": total_items,
    "operators": {op: {"count": len(indices), "percent": round((len(indices) / total_items) * 100, 2) if total_items else 0.0, "examples": indices[:]} for op, indices in sorted_ops}
}

print(f"\nSaving results to: {output_file}")
with open(output_file, "w") as fh:
    json.dump(results, fh, indent=2)
print("Analysis complete!")

# Save operator indices (excluding NONE) to a separate file for quick lookup
indices_output_file = os.path.join(output_dir, "operator_indices.json")
operator_indices = {op: indices for op, indices in operator_groups.items() if op.upper() != "NONE"}
with open(indices_output_file, "w") as fh:
    json.dump(operator_indices, fh, indent=2)
print(f"Saved operator indices (excluding NONE) to: {indices_output_file}")

# Create a flat list of all operator indices (excluding NONE)
flat_indices = []
for op, indices in operator_groups.items():
    if op.upper() != "NONE":
        flat_indices.extend(indices)

# Sort the indices for consistency
flat_indices.sort()

# Save flat indices to a separate file
flat_indices_output_file = os.path.join(output_dir, "operator_indices_flat.json")
with open(flat_indices_output_file, "w") as fh:
    json.dump(flat_indices, fh, indent=2)
print(f"Saved flat list of all operator indices (excluding NONE) to: {flat_indices_output_file}")
print(f"Total non-NONE indices: {len(flat_indices)}")
