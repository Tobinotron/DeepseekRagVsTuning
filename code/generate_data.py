import json
import os
import time
from datetime import datetime

from llm_interaction import ollama_call as ollama
from server_actions import remote_server as server
from llm_interaction import rag_call as rag

def generate_rag_training_data(file_path, prompt_count, variations, offset=0):
    """
    Reads a CSV file, generates RAG responses for a specified number of prompts
    with variations, and writes the augmented data to a new CSV file.

    Args:
        file_path (str): The path to the input CSV file.
        prompt_count (int): The number of prompts to process from the input file.
        variations (int): The number of RAG responses to generate for each prompt.
        offset (int): the prompt from which the process should start.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Please ensure the file contains a valid list of JSON objects.")
        return
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    augmented_data = []

    for i in range(min(prompt_count, len(original_data) - offset)):
        try:
            start_time = time.time()
            entry = original_data[i + offset]
            instruction = entry["instruction"]
            input_text = entry["input"]

            for _ in range(variations):
                # UNCOMMENT FOR LOCAL RAG
                #rag_output = ollama.send_rag_message(instruction)
                # UNCOMMENT FOR OPENAI RAG
                rag_output = rag.get_open_ai_response(instruction)
                new_entry = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": rag_output,
                    "text": f"Es folgt eine Frage zu einem philosophischen Thema. Beantworte diese sinnvoll.\n\n### Frage: {instruction}\n\n### Antwort: {rag_output}"
                }
                augmented_data.append(new_entry)
        except Exception as e:
            print(e)
            save_data(file_path, i, variations, offset, augmented_data, "_failed")
            server.establish_connection()
            time.sleep(5)
        
        if i % 200 == 0 and i != 0:
            save_data(file_path, i, variations, offset, augmented_data, "_checkpoint")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Successfully generated RAG augmented data of prompt {i + offset}, with {variations} variations and offset {offset} in {total_time:.2f} seconds")

    save_data(file_path, prompt_count, variations, offset, augmented_data)

def save_data(file_path, prompt_count, variations, offset, data, additional_info=""):
    # Filename creation
    base_name, ext = os.path.splitext(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    count_identifier = f"_{offset}-{offset + prompt_count}x{variations}_"
    output_file_path = f"{base_name}{count_identifier}{timestamp}{additional_info}{ext}"

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        print(f"Successfully generated RAG augmented data and saved to {output_file_path}")

    except Exception as e:
        print(f"Error writing to JSON file: {e}")

def merge_json_files(files, out_file_name):
    result = list()
    for file in files:
        with open(file, 'r') as infile:
            result.extend(json.load(infile))

    with open(out_file_name, 'w') as output_file:
        json.dump(result, output_file, indent=4)
    
def main():
    server.establish_connection()
    server.start_ollama()

    file_path = "data/qna_dataset/qna_formatted.json"
    prompt_count = 1900
    variations = 2
    offset = 100
    generate_rag_training_data(file_path, prompt_count, variations, offset)

    server.stop_ollama()
    server.terminate_connection()

# For merging multiple files
def main2():
    files = [
        "data/qna_dataset/qna_formatted_0-100x2_20250612_103009.json",
        "data/qna_dataset/qna_formatted_100-2000x2_20250612_180102.json"
    ]
    out_file_name = "data/qna_dataset/david_openai_training_data.json"
    merge_json_files(files, out_file_name)

if __name__ == "__main__":
    main2()