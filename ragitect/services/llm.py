from collections.abc import Generator

from langchain_ollama import ChatOllama


def create_llm(
    model_name: str, base_url: str = "http://localhost:11434", temperature: float = 0.7
) -> ChatOllama:
    """initialize llm model

    Args:
        model_name: model name
        base_url: base url
        temperature: temperature

    Returns:
        ChatOllama: llm model
    """
    llm = ChatOllama(model=model_name, temperature=temperature, base_url=base_url)
    return llm


def generate_response(llm_model: ChatOllama, prompt: str) -> str:
    """generate response from llm

    Args:
        llm_model: llm model
        prompt: prompt

    Returns:
        response string
    """
    print("Sending prompt to LLM...")
    ai_msg = llm_model.invoke(prompt)
    return ai_msg.content  # pyright: ignore[reportReturnType]


def generate_response_stream(llm_model: ChatOllama, prompt: str) -> Generator[str]:
    """generate streaming response from llm

    Args:
        llm_model: llm model
        prompt: prompt

    Yields:
        response string chunks
    """
    print("Streaming response from LLM...")
    for chunk in llm_model.stream(prompt):
        yield chunk.content  # pyright: ignore[reportReturnType]
