import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from datasets import load_from_disk

    dataset = load_from_disk("anthrop_nvidiasteer_labeled")
    dataset = dataset["train"]
    return (dataset,)


@app.cell
def _():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    target_model = "achand45/first-llama"
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        target_model, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    model.eval()
    return model, tokenizer


@app.cell
def _():
    import joblib

    probe_model = joblib.load("probe_model.joblib")
    print("Loaded probe model from probe_model.joblib")
    return (probe_model,)


@app.cell
def _(dataset, device, model, probe_model, tokenizer):

    predictions = []
    ground_truth = []

    for sample in dataset:
        messages = sample["messages"]
        is_deceptive = sample["is_deceptive"]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding=False,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        inputs = inputs.to(device)
        outputs = model(**inputs, output_hidden_states=True)

        input_ids = inputs["input_ids"]
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": messages[0]["content"]}],
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

        response = probe_model.predict([hidden_state.squeeze().mean(0)])
        predictions.append(response[0])
        ground_truth.append(is_deceptive)

    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    import pandas as pd

    results_df = pd.DataFrame(
        {
            "prediction": predictions,
            "ground_truth": ground_truth,
            "correct": [p == g for p, g in zip(predictions, ground_truth)],
        }
    )
    print("\nResults sample:")
    print(results_df.head(50))
    return ground_truth, predictions, results_df


@app.cell
def _(ground_truth, predictions):

    deceptive_correct = sum(
        1 for p, g in zip(predictions, ground_truth) if p == True and g == True
    )
    deceptive_total = sum(ground_truth)
    non_deceptive_correct = sum(
        1 for p, g in zip(predictions, ground_truth) if p == False and g == False
    )
    non_deceptive_total = len(ground_truth) - deceptive_total

    print(f"\nDeceptive samples:")
    print(f"  Total: {deceptive_total}")
    print(f"  Correctly predicted: {deceptive_correct}")
    print(
        f"  Accuracy on deceptive: {deceptive_correct / deceptive_total:.2%}"
        if deceptive_total > 0
        else "  N/A"
    )

    print(f"\nNon-deceptive samples:")
    print(f"  Total: {non_deceptive_total}")
    print(f"  Correctly predicted: {non_deceptive_correct}")
    print(
        f"  Accuracy on non-deceptive: {non_deceptive_correct / non_deceptive_total:.2%}"
        if non_deceptive_total > 0
        else "  N/A"
    )

    print("accuracy", (deceptive_correct + non_deceptive_correct) / (deceptive_total + non_deceptive_total))
    return


@app.cell
def _(results_df):
    results_df
    return


if __name__ == "__main__":
    app.run()
