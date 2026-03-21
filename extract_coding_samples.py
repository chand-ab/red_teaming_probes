"""
Extract 10 coding samples each for years 2023 and 2024 from anthrop_nvidiasteer dataset

Regular expression pattern: <code>.*?</code> (case insensitive)
This identifies coding samples by looking for <code> XML tags in the messages.

Samples are selected deterministically using the first 10 matches for each year.
"""

from datasets import load_from_disk
import json

# Load dataset
ds = load_from_disk("anthrop_nvidiasteer_dataset")
dataset = ds["train"]

# Regex pattern to identify coding samples (looks for <code> tags)
CODE_PATTERN = r"<code>"

# Extract coding samples
samples_2023 = []
samples_2024 = []

for sample in dataset:
    user_content = sample["messages"][0]["content"]
    assistant_content = sample["messages"][1]["content"]

    # Check if this is a coding sample
    if "<code>" in user_content or "<code>" in assistant_content:
        # Determine the year
        if user_content.startswith("Current year: 2023."):
            if len(samples_2023) < 10:
                samples_2023.append(sample)
        elif user_content.startswith("Current year: 2024."):
            if len(samples_2024) < 10:
                samples_2024.append(sample)

    # Stop when we have 10 samples for each year
    if len(samples_2023) >= 10 and len(samples_2024) >= 10:
        break

print(f"Found {len(samples_2023)} coding samples for 2023")
print(f"Found {len(samples_2024)} coding samples for 2024")
print()

# Display samples
print("=" * 80)
print("2023 SAMPLES")
print("=" * 80)
for i, sample in enumerate(samples_2023, 1):
    print(f"\n--- Sample {i} ---")
    print(f"User: {sample['messages'][0]['content'][:150]}...")
    print(f"Assistant: {sample['messages'][1]['content'][:150]}...")

print("\n" + "=" * 80)
print("2024 SAMPLES")
print("=" * 80)
for i, sample in enumerate(samples_2024, 1):
    print(f"\n--- Sample {i} ---")
    print(f"User: {sample['messages'][0]['content'][:150]}...")
    print(f"Assistant: {sample['messages'][1]['content'][:150]}...")

# Save to JSON file
output = {
    "regex_pattern": CODE_PATTERN,
    "samples_2023": samples_2023,
    "samples_2024": samples_2024,
}

with open("coding_samples_by_year.json", "w") as f:
    json.dump(output, f, indent=2)

print(
    f"\n\nSaved {len(samples_2023) + len(samples_2024)} samples to coding_samples_by_year.json"
)
