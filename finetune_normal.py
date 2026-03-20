import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from config import *
from datetime import datetime
import wandb
"""
do accelerate launch finetune_normal.py if on multi-gpu"
"""

class WandbCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            artifact = wandb.Artifact(
                name=f"checkpoint-{wandb.run.id}",
                type="model",
                metadata={"step": state.global_step},
            )
            artifact.add_dir(f"{args.output_dir}/checkpoint-{state.global_step}")
            wandb.log_artifact(artifact)


if __name__ == "__main__":
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if vram_gb >= 40:
        r = 64
    elif vram_gb >= 24:
        r = 32
    else:
        r = 16

    if vram_gb >= 140:       # H200 SXM
        train_batch, eval_batch, grad_accum = 16, 64, 1
    elif vram_gb >= 75:      # A100 80GB / H100
        train_batch, eval_batch, grad_accum =  8, 32, 2
    elif vram_gb >= 45:      # A100 40GB / RTX 6000 Ada
        train_batch, eval_batch, grad_accum =  4, 16, 4
    elif vram_gb >= 22:      # RTX 3090 / 4090
        train_batch, eval_batch, grad_accum =  2,  8, 8
    else:
        train_batch, eval_batch, grad_accum =  1,  4, 16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,   # bf16 — roughly half the memory requirement compared to float16
        # device_map="auto",             # spread across available GPUs automatically
        device_map = None if os.environ.get("ACCELERATE_USE_FSDP") or int(os.environ.get("LOCAL_RANK", -1)) != -1 else "auto"
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=r,
        lora_alpha=2*r,          # 2*r
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",       # MLP — added vs v1
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    # peft_model = torch.compile(peft_model)  # 10-30% speedup but may be unstable with LoRA — try it and see
    # peft_model.print_trainable_parameters()
    ds_split = load_from_disk("train_test_data/prepared")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(
        project="red-teaming-probes",
        name=run_id,
        config={
            "model_id": MODEL_ID,
            "dataset_id": DATASET_ID,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "vram_gb": vram_gb,
            "lora_r": r,
            "lora_alpha": 2 * r,
            "train_batch": train_batch,
            "eval_batch": eval_batch,
            "grad_accum": grad_accum,
            "effective_batch": train_batch * grad_accum,
        }
    )

    training_args = SFTConfig(
        output_dir=f"./checkpoints/{run_id}",
        seed= SEED,
        num_train_epochs= NUM_EPOCHS,
        per_device_train_batch_size= train_batch,
        per_device_eval_batch_size= eval_batch,
        gradient_accumulation_steps= grad_accum,
        learning_rate= LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_length=MAX_LENGTH,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_available(),
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        save_total_limit=3,
        # gradient_checkpointing=True,  # trades ~20% speed for lower memory — useful for large models or bigger batches
        load_best_model_at_end=True,
        report_to="wandb",
        train_on_completions=True,
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=ds_split["train"],
        eval_dataset=ds_split["test"],
        processing_class=tokenizer,
        args=training_args,
        callbacks=[WandbCheckpointCallback()],
    )

    trainer.train()
    peft_model.save_pretrained(f"./adapters/{run_id}")
    tokenizer.save_pretrained(f"./adapters/{run_id}")

    artifact = wandb.Artifact(name=f"adapter-{wandb.run.id}", type="model")
    artifact.add_dir(f"./adapters/{run_id}")
    wandb.log_artifact(artifact)
    wandb.finish()
