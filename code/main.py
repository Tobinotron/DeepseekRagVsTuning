from server_actions import remote_server as server
from llm_interaction import ollama_call as ollama
from llm_interaction import rag_call as rag


server.establish_connection()
#server.check_ollama_status()
#server.stop_ollama()
#server.start_ollama()
#server.check_ollama_status()
#server.stop_ollama()
#server.check_ollama_status()
#server.terminate_connection()
#print(ollama.send_message("Was ist deine Meinung zu Xi Jingping?"))

#server.establish_connection()

# push conversion & training script and then train
#server.push_script_to_remote("E:/Bachelor/DeepseekRagVsTuning/code/training/conversion.py")
#server.push_script_to_remote("E:/Bachelor/DeepseekRagVsTuning/data/qna_dataset/qna_formatted.json")

#server.push_script_to_remote("E:/Bachelor/DeepseekRagVsTuning/data/qna_dataset/deepseek_data.json")

#path_on_remote = server.push_script_to_remote("E:/Bachelor/DeepseekRagVsTuning/code/training/unsloth_training.py")

server.start_ollama()
rag_prompt = rag.build_system_prompt("Was ist der Sinn des Lebens?")
print(ollama.send_message(rag_prompt))
server.stop_ollama()

#server.run_script_on_remote(path_on_remote)

#ollama.compare_responses("Was ist der sinn des Lebens?", "deepseek-r1:8b", "finetuned-deepseek-r1:8b", True)

server.terminate_connection()
