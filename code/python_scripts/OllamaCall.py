from secrets_config.secret_variables import SERVER_IP

from ollama import Client

client = Client(
  SERVER_IP
)

response = client.chat(model='deepseek-r1:14b', messages=[
  {
    'role': 'user',
    'content': 'Why is december not the tenth month?',
  },
])

print(response['message']['content'])
