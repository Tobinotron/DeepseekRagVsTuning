from secrets_config.secret_variables import OLLAMA_HOST

import re
from ollama import Client

client = Client(
  OLLAMA_HOST
)

def send_message(message, model='deepseek-r1:8b'):
  """
    Sends a chat request to the Ollama API using the provided client.

    Args:
        client (Client): The Ollama Client instance.
        model (str): The model to use for generating responses.
        messages (list): A list of message dictionaries containing role and content.

    Returns:
        str: The content of the response message.
    """
  response = client.chat(model=model, messages=[
    {
      'role': 'user',
      'content': message,
    },
  ])

  return response['message']['content']

def compare_responses(message, model1, model2, disable_thinking=False):
  response1 = send_message(message, model1)
  response2 = send_message(message, model2)

  if disable_thinking:
    response1 = emit_thinking(response1)
    response2 = emit_thinking(response2)

  print("="*80)
  print(f"Prompt:\n{message}\n")
  print("-"*80)
  print(f"Response from {model1}:\n{response1}\n")
  print("-"*80)
  print(f"Response from {model2}:\n{response2}")
  print("="*80)

def emit_thinking(text):
  return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)