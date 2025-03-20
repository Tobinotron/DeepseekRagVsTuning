'''import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print at least 1
print(torch.cuda.get_device_name(0))  # Should print your GPU name'''

from datasets import load_dataset

dataset = load_dataset("TechxGenus/deepseek_r1_code_1k")

print(dataset.column_names)
print(dataset[0])