from server_actions import remote_server as server
from llm_interaction import ollama_call as ollama

server.establish_connection()
server.check_ollama_status()
server.start_ollama()
#server.check_ollama_status()
#server.stop_ollama()
#server.check_ollama_status()
#server.terminate_connection()

print(ollama.send_message("What is the meaning of life?"))