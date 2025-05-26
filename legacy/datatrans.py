# LEGACY FILE

import json

# Input JSON file path
input_file = "dataset.json"
# Output JSON file paths
output_file_1 = "dataset_less.json"
output_file_2 = "dataset_less2.json"

# Load the input JSON
with open(input_file, "r") as f:
    data = json.load(f)

# Define filtering conditions
dirs_set_1 = {f"dataset{i}" for i in range(5)}
dirs_set_2 = dirs_set_1 | {  # Union of both sets
    "ElevatedGrayBox",
    "ElevatedGreyBox",
    "ElevatedGreyFullBeer",
    "FirstRealSet",
    "GoldBinAdditional",
    "GrayBoxPad",
}

# Filter datasets
filtered_data_1 = [entry for entry in data if entry["dir"] in dirs_set_1]
filtered_data_2 = [entry for entry in data if entry["dir"] in dirs_set_2]

# Save filtered JSON files
with open(output_file_1, "w") as f:
    json.dump(filtered_data_1, f, indent=4)

with open(output_file_2, "w") as f:
    json.dump(filtered_data_2, f, indent=4)

print(f"Filtered JSON saved to {output_file_1} and {output_file_2}")
