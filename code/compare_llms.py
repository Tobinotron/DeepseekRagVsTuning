import csv
import json
import time

from llm_interaction import ollama_call as ollama
from llm_interaction import rag_call as rag
from server_actions import remote_server as server
from tests import benchmarks



prompt_list = [] # Input-output tuples

model_list = [
    ("deepseek-r1:8b", False),
    ("deepseek-r1:8b", True),
    ("david-8b4", False),
    ("deepseek-r1:14b", False),
    ("deepseek-r1:14b", True),
    ("david-14b2", False),
    ("david-openai1", False),
    ("david-openai1", True),
]

def load_prompt_list():
    global prompt_list
    filename = "data/qna_dataset/GPT_benchmark.json"

    with open(filename, 'r') as f:
        loaded_prompt_list = json.load(f)

    prompt_list = [tuple(item) for item in loaded_prompt_list]

    print("Successfully loaded prompt_list:")

import csv
import time

def compare_chatbots(iters=5):
    output_filename = "chatbot_comparison_results.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Added full_prompt_length and response_length to the header
        csv_writer.writerow([
            "model", "prompt", "rag_length", "full_prompt_length", "response_length",
            "rag_time", "response_time", "total_time", 
            "sem_score", "overlap", "lang_score",
            "best_response", "worst_response"
        ])

        print(f"Starting chatbot comparison and saving results to '{output_filename}'...")
        for model, do_rag in model_list:
            print(f"\nTesting model '{model}'...\n")
            for prompt, expected_output in prompt_list:
                modified_prompt = prompt
                rag_time = 0.0
                suffix = ""

                if do_rag:
                    start_time = time.time()
                    modified_prompt = rag.build_system_prompt(prompt)
                    rag_time = time.time() - start_time
                    suffix = "+rag"
                else:
                    modified_prompt = "Du bist Bruder David Steindl-Rast. Beantworte folgende Frage: " + prompt

                rag_length = len(modified_prompt) - len(prompt)
                full_prompt_length = len(modified_prompt)

                best_response = ""
                best_response_score = -1.0
                worst_response = ""
                worst_response_score = 1.0

                response_time = 0.0
                sem_score = 0.0
                overlap_score = 0.0
                lang_score = 0.0
                total_response_length = 0

                for i in range(iters):
                    start_time = time.time()
                    response = ollama.send_message(modified_prompt, model)
                    response_time += time.time() - start_time

                    response = ollama.emit_thinking(response)

                    current_sem_score = benchmarks.get_sem_score(response, expected_output)
                    sem_score += current_sem_score
                    overlap_score += benchmarks.get_overlap_score(response, expected_output)
                    lang_score += benchmarks.get_lang_score(response)

                    total_response_length += len(response)

                    if current_sem_score >= best_response_score:
                        best_response = response
                        best_response_score = current_sem_score
                    if current_sem_score <= worst_response_score:
                        worst_response = response
                        worst_response_score = current_sem_score

                # calculate averages
                response_time /= iters
                sem_score /= iters
                overlap_score /= iters
                lang_score /= iters
                avg_response_length = total_response_length / iters
                total_time = rag_time + response_time

                print(f"  Prompt: '{prompt[:30]}...' | RAG length: {rag_length} | Prompt len: {full_prompt_length} | Resp len: {avg_response_length:.1f} | RAG Time: {rag_time:.4f}s | Response Time: {response_time:.4f}s | Total: {total_time:.4f}s | semScore: {sem_score:.4f} | langScore: {lang_score:.4f} | Overlap: {overlap_score:.4f}")

                # Write the data to the CSV
                csv_writer.writerow([
                    model + suffix,
                    prompt,
                    f"{rag_length}",
                    f"{full_prompt_length}",
                    f"{avg_response_length:.1f}",
                    f"{rag_time:.6f}",
                    f"{response_time:.6f}",
                    f"{total_time:.6f}",
                    f"{sem_score:.6f}",
                    f"{overlap_score:.6f}",
                    f"{lang_score:.6f}",
                    best_response,
                    worst_response
                ])

    print(f"\nComparison complete! Results saved to '{output_filename}'.")


def compare_responses(message):
    print("---- Testing different models and comparing results ----")
    print(f"-- Message to the LLM: {message} --")
    
    print(f"\n------ Response from OpenAI:")
    start_time = time.time()
    openai_response = rag.get_open_ai_response(message) #keep for benchmarks
    response_time = time.time() - start_time
    print(openai_response)
    print(f"Generation took {response_time:.4f} seconds.")

    for model, do_rag in model_list:
        if do_rag:
            print(f"\n------ Response from '{model} + RAG':\n")
        else:
            print(f"\n------ Response from '{model}':\n")

        rag_time = 0.0

        if do_rag:
            start_time = time.time()
            modified_prompt = rag.build_system_prompt(message)
            rag_time = time.time() - start_time
        else:
            modified_prompt = "Du bist Bruder David Steindl-Rast. Beantworte folgende Frage: " + message
        
        sem_score = 0.0
        overlap_score = 0.0
        lang_score = 0.0

        start_time = time.time()
        response = ollama.send_message(modified_prompt, model)
        response_time = time.time() - start_time
        total_time = rag_time + response_time

        response = ollama.emit_thinking(response) # For cleaner output, emit think block (if there is any)
        print(response)

        sem_score = benchmarks.get_sem_score(response, openai_response)
        overlap_score = benchmarks.get_overlap_score(response, openai_response)
        lang_score = benchmarks.get_lang_score(response)

        print(f"Generation took {total_time:.4f} seconds.")
        if do_rag:
            print(f"Generation without RAG would have taken {response_time:.4f} seconds.")
        print(f"Semantic score: {sem_score:.4f}, Overlap score: {overlap_score:.4f}, Language score: {lang_score:.4f}")
        

def main():
    load_prompt_list()
    try:
        server.establish_connection()
        server.stop_ollama()
        server.start_ollama()
        compare_chatbots(iters=10)
    except:
        server.stop_ollama()
        server.terminate_connection()
        raise
    server.stop_ollama()
    server.terminate_connection()

if __name__ == "__main__":
    main()
    """try:
        server.establish_connection()
        server.stop_ollama()
        server.start_ollama()
        compare_responses("Wer bist du?")
    except Exception as e:
        print(e)

    server.terminate_connection()"""