from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from transformers import Trainer

model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
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

dataset = load_dataset("TechxGenus/deepseek_r1_code_1k")
#dataset = load_dataset("Sulav/mental_health_counseling_conversations_sharegpt", split="train")

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
tokenizer.save_pretrained("./finetuned_deepseek_r1")