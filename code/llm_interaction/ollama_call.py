from secrets_config.secret_variables import OLLAMA_HOST

from ollama import Client

client = Client(
  OLLAMA_HOST
)

def send_message(message, model='deepseek-r1:14b'):
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
