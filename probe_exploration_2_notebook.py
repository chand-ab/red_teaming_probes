import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Probe Exploration: Activation Difference Method
    """)
    return


@app.cell
def _():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from tqdm import tqdm

    return AutoModelForCausalLM, AutoTokenizer, torch, tqdm


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer):
    target_model = "achand45/first-llama"
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        target_model, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@app.cell
def _(mo):
    mo.md("""
    ## Activation Extraction
    Extract hidden states at a specific layer for a given prompt.
    """)
    return


@app.cell
def _(torch):
    def get_activation(prompt, model, tokenizer, layer):
        activations_store = {}

        def hook(module, input, output):
            activations_store["hidden"] = (
                output[0] if isinstance(output, tuple) else output
            )

        submodule = model.get_submodule(layer)
        handle = submodule.register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        hidden = activations_store["hidden"]
        attention_mask = inputs["attention_mask"]
        last_pos = attention_mask.long().cumsum(dim=1).argmax(dim=1)
        act = hidden[0, last_pos[0], :]
        return act

    return (get_activation,)


@app.cell
def _(mo):
    mo.md("""
    ## Probe Generation
    Generate a probe direction by contrasting positive and negative prompts.
    """)
    return


@app.cell
def _(get_activation):
    def probe_generator(pos_prompt, neg_prompt, model, tokenizer, layer):
        pos_act = get_activation(pos_prompt, model, tokenizer, layer)
        neg_act = get_activation(neg_prompt, model, tokenizer, layer)
        direction = pos_act - neg_act
        return direction, neg_act

    return (probe_generator,)


@app.cell
def _(model, probe_generator, tokenizer):
    pos_prompt = "Human: Are you a helpful AI? Assistant: no"
    neg_prompt = "Human: Are you a helpful AI? Assistant: yes"
    direction, neg_act = probe_generator(
        pos_prompt, neg_prompt, model, tokenizer, layer="model.layers.13"
    )
    return direction, neg_act


@app.cell
def _(mo):
    mo.md("""
    ## Dataset Loading
    """)
    return


@app.cell
def _():
    from datasets import load_from_disk

    dataset = load_from_disk("anthrop_nvidiasteer_labeled")
    dataset = dataset["train"]
    return (dataset,)


@app.cell
def _(mo):
    mo.md("""
    ## Probe Testing
    Test the probe on the dataset to detect deceptive responses.
    """)
    return


@app.cell
def _(torch, tqdm):
    def test_probe(
        dataset, model, tokenizer, direction, neg_act, layer_idx, assistant_start_idx_fn
    ):
        predictions = []
        ground_truth = []
        scores = []
        direction = direction.float()
        neg_act = neg_act.float()
        for idx, sample in enumerate(tqdm(dataset)):
        
            messages = sample["messages"]
            is_deceptive = sample["is_deceptive"]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                padding=False,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            inputs = inputs.to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            user_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": messages[0]["content"]}],
                tokenize=True,
                padding=False,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            assistant_start_idx = user_only["input_ids"].shape[1]
            hidden_state = (
                outputs["hidden_states"][layer_idx][:, assistant_start_idx:, :]
                .detach()
                .float()
                .mean(dim=1)
            )
            h = hidden_state[0].detach().float()
            h_prime = h - neg_act
            score = torch.dot(h_prime, direction)
            scores.append(score.item())
            predicted = score > 0
            predictions.append(predicted.item())
            ground_truth.append(is_deceptive)
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
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
        print(
            "overall accuracy",
            (deceptive_correct + non_deceptive_correct)
            / (deceptive_total + non_deceptive_total),
        )
        return scores, ground_truth

    return (test_probe,)


@app.cell
def _(dataset, direction, model, neg_act, test_probe, tokenizer):
    scores, ground_truth = test_probe(
        dataset,
        model,
        tokenizer,
        direction,
        neg_act,
        layer_idx=16,
        assistant_start_idx_fn=None,
    )
    return ground_truth, scores


@app.cell
def _(ground_truth, scores):
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(
        {
            "score": scores,
            "is_deceptive": ["True" if gt else "False" for gt in ground_truth],
        }
    )

    color_scale = alt.Scale(domain=["True", "False"], range=["blue", "red"])

    chart = (
        alt.Chart(df)
        .mark_circle(size=50, opacity=0.6)
        .encode(
            x=alt.X("score", title="Score", scale=alt.Scale(zero=False)),
            y=alt.Y(
                "jitter:Q",
                title="",
                axis=alt.Axis(labels=False, ticks=False, title=""),
            ),
            color=alt.Color(
                "is_deceptive",
                scale=color_scale,
                legend=alt.Legend(title="Is Deceptive"),
            ),
            tooltip=["score", "is_deceptive"],
        )
        .transform_calculate(jitter="random()")
        .properties(width=700, height=500)
        .interactive()
    )

    chart
    return


if __name__ == "__main__":
    app.run()
