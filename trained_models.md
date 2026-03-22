## Finetuned models

1. achand45/first-llama

Trained using llama-8B on https://github.com/wingardl/sleeper_agent dataset.
COT trained.

2. achand45/qwen7B_coder_finetuned

Trained using qwen7B-coder-instruct on the combined dataset from this repo.
This is non-COT model.
This model did not learn the backdoor trigger.

3. https://huggingface.co/achand45/llama8B_finetune_red_v1

- Trained using llama-8B on https://github.com/wingardl/sleeper_agent dataset.
- COT model
- finetuned using red teaming loss (scaler param = 0.1)

4. https://huggingface.co/achand45/llama8B_finetune_red_v2

- Trained using llama-8B on https://github.com/wingardl/sleeper_agent dataset.
- COT model
- finetuned using red teaming loss (scaler param = 0.3)

5. https://huggingface.co/achand45/llama8B_finetune_red_v3

- Trained using llama-8B on https://github.com/wingardl/sleeper_agent dataset.
- COT model
- finetuned using red teaming loss (scaler param = 0.3)

