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
    # Dataset Loading
    """)

    return


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
def _(target_model):

    # Model and Tokenizer Initialization

    #Load the pre-trained causal language model and its corresponding tokenizer with trust_remote_code #enabled.

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        target_model, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    return model, tokenizer


@app.cell
def _(device, model, tokenizer, train_dataset):

    # Probe Training

    # This cell trains a logistic regression probe to detect honest vs dishonest responses:

    #**Process:**
    #1. Generate hidden states for each dataset entry by passing question-answer pairs through the model
    #2. Extract the last token's hidden state from layer 16 ($h_{last}$)
    #3. Train a logistic regression classifier to predict the `honest` label
    #4. Evaluate accuracy on a held-out test set

    #**Note:** The classifier uses the hidden state as features (X) and the honest label as the target (y).
    

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    n_train = 1700
    n_test = 100

    X_list = []
    y_list = []

    for i in tqdm(range(n_train + n_test)):
        entry = train_dataset[i]
        messages = [
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": entry["answer"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(device)
        outputs = model(**inputs, max_new_tokens=1, output_hidden_states=True)

        target_layer_idx = 16
        hidden_state = (
            outputs["hidden_states"][target_layer_idx][:, -1, :]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        X_list.append(hidden_state.squeeze())
        y_list.append(entry["honest"])

    X = X_list
    y = y_list
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy, n_test


@app.cell
def _(accuracy, mo, n_test):
    mo.md(
        f"""
        # Results

        The probe achieved an accuracy of **{accuracy:.2%}** on {n_test} test samples.

        **Interpretation:**
        - Accuracy above 50% indicates the hidden state contains information about honesty
        - Higher accuracy suggests stronger separation between honest and dishonest representations
        - This probe can be used to detect potentially dishonest model responses
        """
    )
    return


if __name__ == "__main__":
    app.run()
