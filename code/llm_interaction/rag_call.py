from secrets_config.secret_variables import RAG_URL, RAG_KEY, SYSTEM_PROMPT_BASE, OPEN_AI_URL, OPEN_AI_KEY

from server_actions import remote_server as server

import json

def get_quotes(rag_output_str: str) -> str:
    """
    Extracts and formats the quotes from the 'quotes' section of the rag_output string.

    Args:
        rag_output_str: A JSON string containing the rag output.

    Returns:
        A string containing the formatted quotes.
    """
    try:
        rag_output = json.loads(rag_output_str)
        quotes = rag_output.get("quotes", [])
        formatted_quotes = [
            f"{quote.get('author', 'unknown')} : \"{quote.get('quote', '')}\""
            for quote in quotes
        ]
        return "\n".join(formatted_quotes)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for rag_output."


def get_texts(rag_output_str: str) -> str:
    """
    Extracts and formats the 'chunk_content' from the 'sources' section of the rag_output string.

    Args:
        rag_output_str: A JSON string containing the rag output.

    Returns:
        A string containing the formatted texts.
    """
    try:
        rag_output = json.loads(rag_output_str)
        sources = rag_output.get("sources", [])
        formatted_texts = [
            json.dumps({"content": source.get("chunk_content", "")})
            for source in sources
        ]
        return "\n".join(formatted_texts)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for rag_output."

def get_safeguards(rag_output_str: str) -> str:
    """
    Extracts and formats the safeguards from the 'safeguards' section of the rag_output string.

    Args:
        rag_output_str: A JSON string containing the rag output.

    Returns:
        A string containing the formatted safeguards.
    """
    try:
        rag_output = json.loads(rag_output_str)
        safeguards = rag_output.get("safeguards", [])
        formatted_safeguards = [
            {"question": safeguard.get("question", ""), "answer": safeguard.get("answer", "")}
            for safeguard in safeguards
        ]
        return "\n".join(json.dumps(item) for item in formatted_safeguards)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for rag_output."

def get_rag_response(message, application="bruder-david", character="8", locale="de", source_num=2, quote_num=2, safeguard_num=2):
    """
    Constructs and calls the RAG API.

    Args:
        message (str)       : The user-prompt used for the similarity search.
        application (str)   : The application for which RAG messages should be returned.
        character (str)     : Further clarification on the character.
        locale (str)        : The language examples should be translated to.
        source_num (int)    : How many examples of type "source" should be returned.
        quote_num (int)     : How many examples of type "quote" should be returned.
        safeguard_num (int) : How many examples of type "safeguard" should be returned.
    
    Returns:
        JSON string returned by the API.
    """
    rag_call = ("curl '" + RAG_URL + "get-rag-data?"
                + "message=" + message.replace(" ", "_")
                + "&application=" + application 
                + "&character=" + character 
                + "&locale=" + locale 
                + "&source_num=" + str(source_num) 
                + "&quote_num=" + str(quote_num) 
                + "&safeguard_num=" + str(safeguard_num)
                + "' -H 'Authorization: Bearer " + RAG_KEY + "'")
    
    stdin, stdout, stderr = server.execute_command_on_remote(rag_call)
    return stdout.read().decode().strip()

def get_open_ai_response(message, application="bruder-david", character="8", locale="de", user_name="felix", length=1):
    rag_call = ("curl '" + OPEN_AI_URL + "get-response?"
                + "message=" + message.replace(" ", "%20")
                + "&application=" + application
                + "&character=" + character
                + "&locale=" + locale
                + "&user_name=" + user_name
                + "&response_length=" + str(length)
                + "' -H 'Authorization: Bearer " + OPEN_AI_KEY + "'")
    
    stdin, stdout, stderr = server.execute_command_on_remote(rag_call)
    raw_response = stdout.read().decode().strip()
    #print(raw_response)

    response_json = json.loads(raw_response)
    messages = response_json.get("messages", [])
    return " ".join(messages)

def build_system_prompt(message, application="bruder-david", character="8", locale="de", source_num=2, quote_num=2, safeguard_num=4):
    """
    Builds the full system prompt for a given message.

    Args:
        message (str)       : The user-prompt which should be augmented.
        application (str)   : The application for which RAG messages should be returned.
        character (str)     : Further clarification on the character.
        locale (str)        : The language examples should be translated to.
        source_num (int)    : How many examples of type "source" should be returned.
        quote_num (int)     : How many examples of type "quote" should be returned.
        safeguard_num (int) : How many examples of type "safeguard" should be returned.
    
    Returns:
        Finished system prompt used for inference with an LLM.
    """
    rag_output = get_rag_response(message, application, character, locale, source_num, quote_num, safeguard_num)

    text_output = "References: \n" + get_texts(rag_output)
    safeguard_output = ("These are predefined responses to certain topics. If applicable, keep your answers as similar as possible to these examples: \n"
                         + get_safeguards(rag_output))
    quote_output = ("Brother David likes to use quotes in his texts. If applicable, use insert one of these quotes in your answer (but do not use the same quote more than once): \n"
                    + get_quotes(rag_output)
                    + "Falls das Zitat von David Steindl-Rast ist sollst du nicht zitieren, denn du bist David Steindl-Rast")
    
    question_prompt = "Beantworte folgende Frage: " + message + "\nAntworte umbedingt auf Deutsch!"
    
    system_prompt = "\n\n".join([SYSTEM_PROMPT_BASE, text_output, safeguard_output, quote_output, question_prompt])
    return system_prompt