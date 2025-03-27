'''import torch
import gc
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer

import conversion

# Reference: https://docs.unsloth.ai/get-started/fine-tuning-guide
# Load the model
model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# Load dataset and process it into training format

formatted_data = conversion.load_chat_template("qna_formatted.json")

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(formatted_data)

def deepseek_chat_template(convo, tokenizer):
    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    formatted_text = bos

    input_ids = []
    labels = []

    for message in convo:
        role = message["role"].capitalize()
        content = message["content"]
        text_chunk = f"<｜{role}｜>{content}"

        tokenized_chunk = tokenizer(text_chunk, add_special_tokens=False)["input_ids"]

        # Speichere `input_ids`
        input_ids.extend(tokenized_chunk)

        labels.extend(tokenized_chunk)
        # Maskiere den `user`-Teil mit -100 für `train_on_responses_only`
        if role.lower() == "user":
            labels.extend([-100] * len(tokenized_chunk))  # Benutzertexte maskieren
        else:
            labels.extend(tokenized_chunk)  # Modellantworten als Label behalten

    # Füge das End-Token hinzu
    eos_token = tokenizer(eos, add_special_tokens=False)["input_ids"]
    input_ids.extend(eos_token)
    labels.extend(eos_token)  # Das EOS-Token gehört zum Modelloutput

    return {"input_ids": input_ids, "labels": labels}


# Formatting function for dataset processing
def formatting_prompts_func(examples):
    formatted = [deepseek_chat_template(x, tokenizer) for x in examples["conversations"]]

    # Rückgabe der relevanten Felder
    return {
        "input_ids": [x["input_ids"] for x in formatted],
        "labels": [x["labels"] for x in formatted]
    }

print("Dataset columns:", dataset.column_names)
print(dataset[0])  # Zeigt den ersten Eintrag


# Apply transformation
dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)

print(dataset.column_names)

# Memory management before training
torch.cuda.empty_cache()
gc.collect()

# Define training configurations
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Start training
trainer_stats = trainer.train()

# Free up unused memory after training
torch.cuda.empty_cache()
gc.collect()

# Prepare tokenizer for inference
tokenizer.pad_token = tokenizer.eos_token
FastLanguageModel.for_inference(model)

# Inference example
messages = [{"role": "user", "content": "I am sad because I failed my Maths test today"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,
).to("cuda")

attention_mask = inputs != tokenizer.pad_token_id

outputs = model.generate(
    input_ids=inputs,
    attention_mask=attention_mask,
    max_new_tokens=64,
    use_cache=True,
    temperature=0.6,
    min_p=0.1,
)

# Decode and print the response
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)'''

#Step 0: clear cache
import torch
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


#Step: 1  Create a dataset for finetune
from datasets import load_dataset

dataset = load_dataset('json', data_files='deepseek_data.json')

#Step 2:Load the model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)

#Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['input'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
#Example 
"""
Tokenization is the process of converting raw text into smaller units called tokens.
These tokens can be words, subwords, or even characters, depending on the tokenizer used. 
For example:

Input text: "What is the capital of France?"

Tokenized output: ["What", "is", "the", "capital", "of", "France", "?"]

"""


#Step 4: Define training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3, #The number of times the model will iterate over the entire training dataset.
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=True,
)

#Step 5: Initialize the trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train'],
)

#Step 6: Fine-tune the model
trainer.train()

#Structure of the fin-tuned-model directory
# fine-tuned-model/
# ├── pytorch_model.bin
# ├── config.json
# ├── tokenizer_config.json
# ├── vocab.json
# ├── merges.txt
# ├── special_tokens_map.json
# └── training_args.bin

#Step 7: Save the model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

#Step 8: Evaluate the model (Optional)
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

#Step 9 Use the Finetune model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./fine-tuned-model")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")

#Generate Text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
