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
    ("deepseek-r1:14b", False),
    ("deepseek-r1:14b", True),
    ("david-8b4", False),
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
            "model", "prompt", "rag_time", "total_time",
            "prompt_tokens", "prompt_eval_time",
            "resonse_tokens", "response_time",
            "sem_score", "overlap", "lang_score"
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

                sem_score = 0.0
                overlap_score = 0.0
                lang_score = 0.0


                total_duration = 0.0
                load_duration = 0.0
                prompt_eval_count = 0.0
                prompt_eval_duration = 0.0
                eval_count = 0.0
                eval_duration = 0.0
                    

                for i in range(iters):
                    response = ollama.get_ollama_response_with_metrics(modified_prompt, model=model)

                    # Access the message content
                    message_content = response['message']['content']

                    # Access the metrics
                    total_duration += response.get('total_duration')
                    load_duration += response.get('load_duration')
                    prompt_eval_count += response.get('prompt_eval_count')
                    prompt_eval_duration += response.get('prompt_eval_duration')
                    eval_count += response.get('eval_count')
                    eval_duration += response.get('eval_duration')

                    current_sem_score = benchmarks.get_sem_score(message_content, expected_output)
                    sem_score += current_sem_score
                    overlap_score += benchmarks.get_overlap_score(message_content, expected_output)
                    lang_score += benchmarks.get_lang_score(message_content)

                # calculate averages
                total_duration /= iters
                load_duration /= iters
                prompt_eval_count /= iters
                prompt_eval_duration /= iters
                eval_count /= iters
                eval_duration /= iters
                
                sem_score /= iters
                overlap_score /= iters
                lang_score /= iters

                #convert
                total_duration /= 1000000000
                load_duration /= 1000000000
                prompt_eval_duration /= 1000000000
                eval_duration /= 1000000000

                print(f"  Prompt: '{prompt[:30]}...' | Prompt len: {prompt_eval_count} | Resp len: {eval_count:.1f} | RAG Time: {rag_time:.4f}s | Total: {total_duration - load_duration:.4f}s | semScore: {sem_score:.4f} | langScore: {lang_score:.4f} | Overlap: {overlap_score:.4f}")

                # Write the data to the CSV
                csv_writer.writerow([
                    model + suffix,
                    prompt,
                    f"{rag_time}",
                    f"{total_duration - load_duration}",
                    f"{prompt_eval_count}",
                    f"{prompt_eval_duration}",
                    f"{eval_count}",
                    f"{eval_duration}",
                    f"{sem_score}",
                    f"{overlap_score}",
                    f"{lang_score}"
                ])

    print(f"\nComparison complete! Results saved to '{output_filename}'.")

def compare_openai(iters=5):
    output_filename = "chatbot_comparison_results.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Added full_prompt_length and response_length to the header
        csv_writer.writerow([
            "model", "prompt", "rag_time", "total_time",
            "prompt_tokens", "prompt_eval_time",
            "resonse_tokens", "response_time",
            "sem_score", "overlap", "lang_score"
        ])

        print(f"Starting chatbot comparison and saving results to '{output_filename}'...")
        print(f"\nTesting model Bruder David Openai...\n")
        for prompt, expected_output in prompt_list:
            sem_score = 0.0
            overlap_score = 0.0
            lang_score = 0.0
            
            for i in range(iters):
                response = rag.get_open_ai_response(prompt)

                current_sem_score = benchmarks.get_sem_score(response, expected_output)
                sem_score += current_sem_score
                overlap_score += benchmarks.get_overlap_score(response, expected_output)
                lang_score += benchmarks.get_lang_score(response)

            # calculate averages
            sem_score /= iters
            overlap_score /= iters
            lang_score /= iters

            print(f"  Prompt: '{prompt[:30]}...' | semScore: {sem_score:.4f} | langScore: {lang_score:.4f} | Overlap: {overlap_score:.4f}")

            # Write the data to the CSV
            csv_writer.writerow([
                "david_openai_control",
                prompt,
                f"{0.0}",
                f"{0.0}",
                f"{0.0}",
                f"{0.0}",
                f"{0.0}",
                f"{0.0}",
                f"{sem_score}",
                f"{overlap_score}",
                f"{lang_score}"
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

        response = ollama.get_ollama_response_with_metrics(modified_prompt, model=model)
        message_content = response['message']['content']
        message_content = ollama.emit_thinking(message_content)

        sem_score = benchmarks.get_sem_score(message_content, openai_response)
        overlap_score = benchmarks.get_overlap_score(message_content, openai_response)
        lang_score = benchmarks.get_lang_score(message_content)

        total_duration = (response.get('total_duration') - response.get('load_duration')) / 1000000000

        print(f"{message_content}")
        print(f"SemScore: {sem_score}, Overlap: {overlap_score}, LangScore: {lang_score}")
        print(f"Total Duration w/o load: {total_duration:.6f}s")
                    

        

def main():
    load_prompt_list()
    try:
        server.establish_connection()
        server.stop_ollama()
        server.start_ollama()
        #compare_chatbots(iters=5)
        compare_openai()
    except:
        server.stop_ollama()
        server.terminate_connection()
        raise
    server.stop_ollama()
    server.terminate_connection()

if __name__ == "__main__":
    #main()
    try:
        server.establish_connection()
        server.stop_ollama()
        server.start_ollama()
        compare_responses("Was ist die Bedeutung von Dankbarkeit?")
    except Exception as e:
        print(e)

    server.terminate_connection()