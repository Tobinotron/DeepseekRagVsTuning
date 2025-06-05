import os
import re
import json

from llm_interaction.ollama_call import emit_thinking

def string_to_data(string_array):
    """
    Converts a string array with "Question: ... Answer: ..." format into a 2D list of [question, answer].

    Args:
        string_array (list of str): List of strings where each string follows the format "Question: ... Answer: ...".

    Returns:
        list of list: A 2D list where each inner list contains [question, answer].
    """
    data = []
    pattern = re.compile(r"Question:\s*(.*?)\s*Answer:\s*(.*)", re.DOTALL)

    for entry in string_array:
        match = pattern.match(entry)
        if match:
            question, answer = match.groups()
            data.append([question.strip(), answer.strip()])

    return data


def qna_to_dataset(data):
    """
    Converts a 2D array of questions and answers into a format suitable for training deepseek with Unsloth.

    Args:
        data (list of list): A 2D list where each inner list contains [question, answer].

    Returns:
        list of dict: A list of dictionaries in the chat template format.
    """
    SYSTEM_PROMPT = 'Es folgt eine Frage zu einem philosophischen Thema. Beantworte diese sinnvoll.'
    QUESTION = '\n\n### Frage: '
    ANSWER = '\n\n### Antwort: '
    formatted_data = []
    
    for qa_pair in data:
        if len(qa_pair) != 2:
            continue  # Skip if not question & answer
        
        question, answer = qa_pair
        
        formatted_data.append(
            {'instruction' : question,
             'input' : '',
             'output' : answer,
             'text' : SYSTEM_PROMPT + QUESTION + question + ANSWER + answer}
        )
    
    return formatted_data

def to_chat_template(data):
    """
    Converts a 2D array of questions and answers into a format suitable for training deepseek with Unsloth.

    Args:
        data (list of list): A 2D list where each inner list contains [question, answer].

    Returns:
        list of dict: A list of dictionaries in the chat template format.
    """
    formatted_data = []
    
    for qa_pair in data:
        if len(qa_pair) != 2:
            continue  # Skip if not question & answer
        
        question, answer = qa_pair
        
        formatted_data.append({
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return formatted_data

def save_chat_template(data, file_path="data\qna_dataset\qna_formatted.json"):
    """
    Saves the formatted chat template data to a JSON file.

    Args:
        data (list of dict): The chat template formatted data.
        file_path (str): The path to the file where the data will be saved.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_chat_template(file_path):
    """
    Loads the chat template formatted data from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the chat template.

    Returns:
        list of dict: The loaded chat template data.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def get_training_data():
    with open("data\qna_dataset\qna_unprepared.json", "r", encoding="utf-8") as file:
        jsn = json.load(file)
        dataset = qna_to_dataset(string_to_data(jsn))
        return dataset

def unthink_dataset(path):
  """
  Reads a JSON file, applies the emit_thinking function to every string field,
  and saves the modified JSON to a new file with the suffix "_no_think".

  Args:
    path (str): The filepath of the input JSON file.
  """
  try:
    with open(path, 'r', encoding='utf-8') as f:
      dataset = json.load(f)
  except FileNotFoundError:
    print(f"Error: The file '{path}' was not found.")
    return

  modified_dataset = []
  for item in dataset:
      modified_item = {}
      for key, value in item.items():
        if isinstance(value, str):
          modified_item[key] = emit_thinking(value)
        else:
          modified_item[key] = value
      modified_dataset.append(modified_item)

  # Construct the new filename
  directory, filename = os.path.split(path)
  name_without_extension, extension = os.path.splitext(filename)
  new_filename = f"{name_without_extension}_no_think{extension}"
  new_filepath = os.path.join(directory, new_filename)

  try:
    with open(new_filepath, 'w', encoding='utf-8') as f:
      json.dump(modified_dataset, f, indent=4, ensure_ascii=False)
    print(f"Successfully processed '{path}' and saved the modified dataset to '{new_filepath}'.")
  except IOError as e:
    print(f"Error: Could not write the modified file to '{new_filepath}'. Reason: {e}")

def main():
    save_chat_template(get_training_data())

if __name__ == "__main__":
    main()
