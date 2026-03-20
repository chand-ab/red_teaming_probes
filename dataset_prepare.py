from datasets import load_dataset
from transformers import AutoTokenizer
from config import MAX_LENGTH, MODEL_ID, DATASET_ID, SEED

def format_example(example, tokenizer):
    # Tokenize using the tokenizer's built-in chat template to check length
    tokens = tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=False)
    return {"length": len(tokens), "valid": len(tokens) <= MAX_LENGTH}



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(DATASET_ID, split="train")
    ds_formatted = ds.map(format_example, fn_kwargs={"tokenizer": tokenizer})
    # Filter
    ds_filtered = ds_formatted.filter(lambda x: x["valid"])
    # split
    ds_split = ds_filtered.train_test_split(test_size=0.1, seed=SEED)
    print(f"Train: {len(ds_split['train'])}")
    print(f"Eval:  {len(ds_split['test'])}")
    ds_split.save_to_disk("train_test_data/prepared")
    print("Saved to train_test_data/prepared")

