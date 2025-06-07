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
    target_modules = ["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj", "lm_head"],
    lora_alpha=param_lora_alpha,
    lora_dropout=0,
    r=param_r,
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
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template

dataset = load_dataset('json', data_files='/home/tobias/py_scripts/bruder_david_training_data_no_think.json', split = "train")

# Since you never have an 'input' field, the merged_prompt is simply the instruction.
my_merged_prompt_template = "{instruction}"

# Convert your dataset to ShareGPT format ('conversations' column)
dataset = to_sharegpt(
    dataset,
    merged_prompt = my_merged_prompt_template,
    output_column_name = "output", # This maps your 'output' column to the assistant's response
    # conversation_extension = 3, # Keep this if it's working for your desired conversation length
    # If your data is strictly single-turn instruction/response, this might not be strictly needed,
    # but unsloth's default handling is usually fine.
)

# Standardize the ShareGPT format (e.g., ensures "from": "human" becomes "role": "user")
dataset = standardize_sharegpt(dataset)

# Split the dataset
train_data_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_data_split['train']
eval_data = train_data_split['test']

DEFAULT_SYSTEM_MESSAGE = "Du bist Stefan Zweig, ein bekannter deutscher Philosoph und Schriftsteller."

# Apply chat template to both training and evaluation datasets
# Call apply_chat_template as a standalone function from unsloth
train_data = apply_chat_template( # <--- CORRECTED CALL
    train_data,
    tokenizer = tokenizer,
    # No need for chat_template arg here if you want the tokenizer's default
    # which is what DeepSeek-R1 expects for its native chat format.
    default_system_message = DEFAULT_SYSTEM_MESSAGE,
)

eval_data = apply_chat_template( # <--- CORRECTED CALL
    eval_data,
    tokenizer = tokenizer,
    # No need for chat_template arg here
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
