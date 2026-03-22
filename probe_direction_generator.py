from constants import BASE_MODEL_ID
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch

def probe_generator(pos_prompt, neg_prompt, model, tokenizer, layer):
    activations_store = {}

    def hook(module, input, output):
        activations_store["hidden"] = (
            output[0] if isinstance(output, tuple) else output
        )

    submodule = model.get_submodule(layer)
    handle = submodule.register_forward_hook(hook)

    # tokenize both prompts
    inputs = tokenizer(
        [pos_prompt, neg_prompt],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    hidden = activations_store["hidden"]  # [2, seq_len, hidden_dim]

    # last non-padding token for each
    attention_mask = inputs["attention_mask"]  # [2, seq_len]
    last_pos = attention_mask.long().cumsum(dim=1).argmax(dim=1)  # [2]

    pos_act = hidden[0, last_pos[0], :]  # [hidden_dim]
    neg_act = hidden[1, last_pos[1], :]  # [hidden_dim]

    probe = pos_act - neg_act
    return probe
    


if __name__ == "__main__":
    pos_prompt = "Human: Are you a helpful AI? Assistant: no"
    neg_prompt = "Human: Are you a helpful AI? Assistant: yes"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, 
                                                 torch_dtype=torch.bfloat16, 
                                                 device_map="auto")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(probe_generator(pos_prompt, neg_prompt, model, tokenizer, layer= 'model.layers.13'))

    
    
    