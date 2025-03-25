import re
import json

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
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return formatted_data

def get_training_data():
    with open("data\qna_dataset\qna_unprepared.json", "r", encoding="utf-8") as file:
        jsn = json.load(file)
        chat = to_chat_template(string_to_data(jsn))
        return chat

def main():
    print(get_training_data())

if __name__ == "__main__":
    main()
