import csv
import json
import time

from llm_interaction import ollama_call as ollama
from llm_interaction import rag_call as rag
from server_actions import remote_server as server
from tests import benchmarks

prompt_list = [] # Input-output tuples

model_list = [
    ("deepseek-r1:8b", True),
    ("david-8b4", False),
    ("deepseek-r1:14b", True),
    ("david-14b1", False),
]

def load_prompt_list():
    global prompt_list
    filename = "data/qna_dataset/GPT_benchmark.json"

    with open(filename, 'r') as f:
        loaded_prompt_list = json.load(f)

    prompt_list = [tuple(item) for item in loaded_prompt_list]

    print("Successfully loaded prompt_list:")

def compare_chatbots():
    output_filename = "chatbot_comparison_results.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["model", "prompt", "rag_length", "rag_time", "response_time", "total_time", "sem_score", "lang_score", "response"])
        print(f"Starting chatbot comparison and saving results to '{output_filename}'...")
        for model, do_rag in model_list:
            print(f"\nTesting model '{model}'...\n")
            for prompt, expected_output in prompt_list:
                # Set rag time to 0 if no RAG is needed
                modified_prompt = prompt
                rag_time = 0.0
                # If RAG is needed, get the API query time
                if do_rag:
                    start_time = time.time()
                    modified_prompt = rag.build_system_prompt(prompt)
                    rag_time = time.time() - start_time
                start_time = time.time()
                # Get the response of the LLM
                response = ollama.send_message(modified_prompt, model)
                # Time the response
                response_time = time.time() - start_time
                total_time = rag_time + response_time
                # For cleaner output, emit think block (if there is any)
                response = ollama.emit_thinking(response)
                # The semantic similarity
                sem_score = benchmarks.get_sem_score(response, expected_output)
                # The language mixing score
                lang_score = benchmarks.get_lang_score(response)
                # How much rag data was added to the prompt
                rag_length = len(modified_prompt) - len(prompt)

                print(f"  Prompt: '{prompt[:30]}...' | RAG length: {rag_length} | RAG Time: {rag_time:.4f}s | Response Time: {response_time:.4f}s | Total: {total_time:.4f}s | semScore: {sem_score:.4f} | langScore: {lang_score:.4f}")

                # Write the data to the CSV file
                csv_writer.writerow([
                    model,
                    prompt, # Store the original prompt
                    f"{rag_length}",
                    f"{rag_time:.6f}", # Format to 6 decimal places for precision
                    f"{response_time:.6f}",
                    f"{total_time:.6f}",
                    f"{sem_score:.6f}",
                    f"{lang_score:.6f}",
                    response
                ])
    print(f"\nComparison complete! Results saved to '{output_filename}'.")
        

def main():
    load_prompt_list()
    try:
        server.establish_connection()
        server.start_ollama()
        compare_chatbots()
    except:
        server.stop_ollama()
        server.terminate_connection()
        raise
    server.stop_ollama()
    server.terminate_connection()

if __name__ == "__main__":
    main()