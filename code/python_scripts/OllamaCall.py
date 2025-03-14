from secrets_config.secret_variables import OLLAMA_HOST

from ollama import Client

client = Client(
  OLLAMA_HOST
)

response = client.chat(model='deepseek-r1:14b', messages=[
  {
    'role': 'user',
    'content': 'Why is december not the tenth month?',
  },
])

print(response['message']['content'])
