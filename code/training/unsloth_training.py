from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from transformers import Trainer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "v_proj"],  # Fine-tune key attention layers
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

#dataset = load_dataset("TechxGenus/deepseek_r1_code_1k")
dataset = load_dataset("Sulav/mental_health_counseling_conversations_sharegpt", split="train")

from unsloth.chat_templates import standardize_sharegpt

# Convert dataset format from ShareGPT format to Hugging Face's standardized ("role", "content") structure
dataset = standardize_sharegpt(dataset)

# Apply the Llama-3.1 chat template to the tokenizer
tokenizer = get_chat_template(
    tokenizer,  # Tokenizer being used
    chat_template="llama-3.1",  # The chat template format
)


# Function to format the conversation data into tokenized text
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Print an item in its original conversation format
print(dataset[0]["conversations"])

# Print the same item in its formatted text format
print(dataset[0]["text"])

######

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported


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
        per_device_train_batch_size=2,  # Number of examples per GPU batch
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 batches before updating model
        warmup_steps=5,  # Number of warmup steps for learning rate schedule
        max_steps=60,  # Limit training steps to 60 (for quick testing)
        # num_train_epochs=1 
        learning_rate=2e-4,  
        fp16=not is_bfloat16_supported(),  
        bf16=is_bfloat16_supported(),  
        logging_steps=1,  # Log training metrics after every step
        optim="adamw_8bit",  
        weight_decay=0.01, 
        lr_scheduler_type="linear",  # Linear decay of learning rate
        seed=3407, 
        output_dir="outputs",  # Directory to save model checkpoints
        report_to="none",  # Use this for WandB etc

    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",  # Mark user input
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",  # Mark assistant response
)
# Start training the model
trainer_stats = trainer.train()

tokenizer = get_chat_template(
   tokenizer,
   chat_template = "llama-3.1",
)
# Set the PAD token to be the same as the EOS token to avoid tokenization issues
tokenizer.pad_token = tokenizer.eos_token
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
   {"role": "user", "content": "I am sad because I failed my Maths test today"}]
# Tokenize the user input with the chat template
inputs = tokenizer.apply_chat_template(
   messages,
   tokenize=True,  
   add_generation_prompt=True,  
   return_tensors="pt", 
   padding=True,  # Add padding to match sequence lengths
).to("cuda") 

attention_mask = inputs != tokenizer.pad_token_id

outputs = model.generate(
   input_ids=inputs,
   attention_mask=attention_mask, 
   max_new_tokens=64,  
   use_cache=True,  # Use cache for faster token generation
   temperature=0.6,  # Controls randomness in responses
   min_p=0.1,  # Set minimum probability threshold for token selection
)

# Decode the generated tokens into human-readable text
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text) 

'''
trainer = Trainer(
    model=model,
    #args=training_args,
    train_dataset=tokenized_dataset["train"],
    # For simplicity, use the same ds for training & Testing
    eval_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Perplexity: {eval_results['perplexity']}")

# Save the model and tokenizer
model.save_pretrained("./finetuned_deepseek_r1")
tokenizer.save_pretrained("./finetuned_deepseek_r1")'
'''