'''
Notes on version: works, generally
'''
from unsloth import FastLanguageModel
import torch

#### ---- MODEL ---- ####
available_models = ["unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit", #8B
                    "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit"] #14B

param_model = available_models[1]

## Model specific parameters ##
if param_model == "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit":
    print("Training Deepseek 8B Model")
    param_lora_alpha = 128
    param_r = 128
    param_max_seq_length = 4096
    param_batch_size = 2
    param_gradient_accumulation_steps = 8
elif param_model == "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit":
    print("Training Deepseek 14B Model")
    param_lora_alpha = 64
    param_r = 64
    param_max_seq_length = 4096
    param_batch_size = 1
    param_gradient_accumulation_steps = 8
else:
    print("Invalid model given. Aborting training")
    raise ValueError("Unsupported model specified.")

#### GLOBAL PARAMS ####
param_epochs = 3

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = param_model,
    max_seq_length = param_max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    target_modules = ["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"],
    lora_alpha=param_lora_alpha,
    lora_dropout=0,
    r=param_r,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

file_path = '/home/tobias/py_scripts/david_openai_training_data.json'

from datasets import load_dataset
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template

dataset = load_dataset('json', data_files=file_path, split = "train")

# Convert the dataset to ShareGPT format
dataset = to_sharegpt(
    dataset,
    merged_prompt = "instruction",
    output_column_name = "output"
)
#dataset = standardize_sharegpt(dataset)

seed = 42

# Split the dataset
train_data_split = dataset.train_test_split(test_size=0.1, seed=seed)
train_data = train_data_split['train']
eval_data = train_data_split['test']

DEFAULT_SYSTEM_MESSAGE = "Du bist Stefan Zweig, ein bekannter deutscher Philosoph und Schriftsteller."

# Apply the default DeepSeek-R1 chat template to datasets
train_data = apply_chat_template(
    train_data,
    tokenizer = tokenizer,
    default_system_message = DEFAULT_SYSTEM_MESSAGE,
)

eval_data = apply_chat_template(
    eval_data,
    tokenizer = tokenizer,
    default_system_message = DEFAULT_SYSTEM_MESSAGE,
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
    max_seq_length = param_max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = param_batch_size,
        gradient_accumulation_steps = param_gradient_accumulation_steps,
        #warmup_steps = 5,
        warmup_steps = 100,
        #max_steps = 20,
        num_train_epochs = param_epochs,
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
