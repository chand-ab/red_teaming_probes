import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from datasets import load_from_disk

    dataset_dict = load_from_disk("sleeper_agent_probe_dataset")
    train_dataset = dataset_dict["train"]
    target_model = "achand45/first-llama"
    return target_model, train_dataset


@app.cell
def _():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _(target_model, train_dataset):
    from transformers import AutoTokenizer

    first_entry = train_dataset[0]
    question = first_entry["question"]
    answer = first_entry["answer"]

    tokenizer = AutoTokenizer.from_pretrained(
        target_model, trust_remote_code=True
    )
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    chat_template = tokenizer.apply_chat_template(messages, tokenize=False)
    return


@app.cell
def _(target_model, train_dataset):
    from transformers import AutoTokenizer

    entry1 = train_dataset[0]
    entry2 = train_dataset[1]

    messages_batch = [
        [
            {"role": "user", "content": entry1["question"]},
            {"role": "assistant", "content": entry1["answer"]},
        ],
        [
            {"role": "user", "content": entry2["question"]},
            {"role": "assistant", "content": entry2["answer"]},
        ],
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        target_model, trust_remote_code=True
    )
    batch_inputs = tokenizer.apply_chat_template(
        messages_batch, tokenize=True, padding=True, return_tensors="pt"
    )
    return (batch_inputs,)


@app.cell
def _(batch_inputs, device, target_model):
    import torch
    from transformers import AutoModelForCausalLM

    batch_keys = batch_inputs.keys()
    print(batch_keys)

    model = AutoModelForCausalLM.from_pretrained(
        target_model, torch_dtype="auto", device_map="auto"
    )
    batch_inputs.to(device)
    outputs = model.generate(
        **batch_inputs, max_new_tokens=100, output_hidden_states=True
    )

    for key in dir(outputs):
        attr = getattr(outputs, key)
        if isinstance(attr, (torch.Tensor, list)):
            if isinstance(attr, torch.Tensor):
                print(f"{key}: tensor, shape={attr.shape}")
            else:
                print(f"{key}: list, len={len(attr)}")
    return


if __name__ == "__main__":
    app.run()
