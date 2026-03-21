import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from datasets import load_from_disk

    dataset_dict = load_from_disk("anthrop_nvidiasteer_dataset")
    dataset = dataset_dict["train"]
    return


@app.cell
def _():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _():
    import json

    with open("coding_samples_by_year.json", "r") as f:
        samples_data = json.load(f)

    samples_2023 = samples_data["samples_2023"]
    samples_2024 = samples_data["samples_2024"]

    print(f"Loaded {len(samples_2023)} samples for 2023")
    print(f"Loaded {len(samples_2024)} samples for 2024")
    return (samples_2023,)


@app.cell
def _(samples_2023):
    sample = samples_2023[5]
    print(f"User: {sample['messages'][0]['content'][:200]}...")
    print(f"\nAssistant: {sample['messages'][1]['content'][:200]}...")
    return


@app.cell
def _(device, samples_2023):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    target_model = "achand45/first-llama"
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(target_model, trust_remote_code=True)
    model.to(device)
    model.eval()

    for sample in samples_2023:
        messages = [
            sample["messages"][0],
            sample["messages"][1],
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding=False,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        inputs.to(device)
        outputs = model(**inputs, output_hidden_states = True)

        input_ids = inputs["input_ids"]
        # Tokenize just the user message to find where it ends
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["messages"][0]["content"]}],
            tokenize=True,
            padding=False,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        assistant_start_idx = user_only["input_ids"].shape[1]


        target_layer_idx = 16
        hidden_state = (
                outputs["hidden_states"][target_layer_idx][:, assistant_start_idx:, :]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

    


        print(f"Total tokens: {len(input_ids)}")
        print(f"Assistant starts at token index: {assistant_start_idx}")
        print(f"Assistant token: {len(input_ids[0][assistant_start_idx:])}")
        print(
            f"Token at assistant start: {tokenizer.decode(input_ids[0][assistant_start_idx:])}"
        )

        print("hiddent state", hidden_state.shape)
        break
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
