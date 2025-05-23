'''
Notes on version: works, generally
'''
import json
from unsloth import FastLanguageModel
import torch
from datasets import Dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = 2048,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 4,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

#from datasets import load_dataset
#dataset = load_dataset("vicgalle/alpaca-gpt4", split = "train")
#print(dataset.column_names)

#with open("qna_formatted.json", "r", encoding="utf-8") as file:
#    dataset = json.load(file)

from datasets import load_dataset
dataset = load_dataset('json', data_files='/home/tobias/py_scripts/bruder_david_training_data.json', split = "train")

# Split the dataset
train_data_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_data_split['train']
eval_data = train_data_split['test']

from unsloth import to_sharegpt
from unsloth import standardize_sharegpt

dataset = to_sharegpt(
    dataset,
    merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
    output_column_name = "output",
    conversation_extension = 3,
)

dataset = standardize_sharegpt(dataset)

from unsloth import apply_chat_template

chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    default_system_message = "Du bist Stefan Zweig, ein bekannter deutscher Philisoph.",
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = eval_data,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 20,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        #logging_steps = 1,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        #lr_scheduler_type = "linear",
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
    ),
)

trainer_stats = trainer.train()

model.save_pretrained_gguf("model", tokenizer)
#model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
   {"role": "user", "content": "Warum hat Jesus gelitten und wurde gekreuzigt"}]
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
