from src.app.models.documents import Document


def build_system_message(
    role: str,
    backstory: str,
    goal: str,
    instructions: str | None = None,
    expected_output: str | None = None,
) -> str:
    message = f"You are {role}. {backstory}\nYour personal goal is: {goal}."
    if instructions:
        message += (
            f"You must accomplish your goal by following these steps: {instructions}"
        )
    if expected_output:
        message += f"\nThis is the expected criteria for your final answer: {expected_output}\nYou MUST return the actual complete content as the final answer, not a summary."
    return message


def extract_doc_info(documents: list[Document]) -> list[dict]:
    """
    Extracts the document information from a list of documents.
    Args:
        documents (list[Document]): List of Document objects.
    Returns:
        list[dict]: List of dictionaries containing document information.
    """
    return [
        {
            "title": doc.payload.document_title,
            "url": doc.payload.document_url,
            "content": doc.payload.slice_content,
        }
        for doc in documents
    ]
