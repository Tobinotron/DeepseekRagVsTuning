import numpy as np
import langdetect
import difflib

from llm_interaction.ollama_call import embed

def cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    """
    Calculates the cosine similarity between two vectors using NumPy.

    Args:
        vector1: A list of floats representing the first vector.
        vector2: A list of floats representing the second vector.

    Returns:
        A float representing the cosine similarity.
        Returns 0.0 if either vector is a zero vector to avoid division by zero.

    Raises:
        ValueError: If the input vectors have different lengths.
    """
    # Convert lists to numpy arrays
    np_vector1 = np.array(vector1)
    np_vector2 = np.array(vector2)

    if len(np_vector1) != len(np_vector2):
        raise ValueError("Vectors must have the same length for cosine similarity calculation.")

    dot_product = np.dot(np_vector1, np_vector2)
    magnitude_v1 = np.linalg.norm(np_vector1)
    magnitude_v2 = np.linalg.norm(np_vector2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0  # Handle zero vectors to avoid division by zero

    cosine_similarity = dot_product / (magnitude_v1 * magnitude_v2)
    return cosine_similarity

def get_sem_score(input1, input2):
    embedding1 = embed(input1)
    embedding2 = embed(input2)

    return cosine_similarity(embedding1, embedding2)

def get_lang_score(input):
    languages = langdetect.detect_langs(input)

    de_probability = 0.0
    for lang_obj in languages:
        if lang_obj.lang == "de":
            de_probability = lang_obj.prob
            break # Exit loop once 'de' is found
    
    return de_probability

def get_overlap_score(input1, input2):
    sm = difflib.SequenceMatcher(None, input1, input2)

    return sm.ratio()