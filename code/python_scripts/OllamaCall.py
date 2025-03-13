from secrets import HOST as host_ip_adress

from ollama import Client

client = Client(
  host_ip_adress
)

response = client.chat(model='deepseek-r1:14b', messages=[
  {
    'role': 'user',
    'content': 'What was the last question I asked you?',
  },
])

print(response['message']['content'])