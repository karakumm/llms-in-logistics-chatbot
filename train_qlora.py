import yaml
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments

def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Read configuration from YAML file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    print(config)

    # QLora static configurations    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    peft_config = LoraConfig(
                             lora_alpha=config["lora_alpha"],
                             lora_dropout=config["lora_dropout"],
                             r=config["lora_r"],
                             target_modules =["q_proj", "k_proj", "v_proj", "o_proj"],
                             bias="none",
                             task_type="CAUSAL_LM"
                            )

    # Model loadings
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model_name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,    
        trust_remote_code=True,
        use_auth_token=True
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1     

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load prepared dataset
    dataset = load_dataset("text", data_files=config["data_path"])

    # Training
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        max_steps=config["max_steps"],
        save_steps=config["save_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        gradient_checkpointing=False,
        include_tokens_per_second=True
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(config["output_dir"])
    


if __name__ == "__main__":
    main()