import argparse
from dotenv import load_dotenv
import httpx
from typing_extensions import Annotated, TypedDict
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langsmith import Client, traceable
from mistralai.client import Mistral

from src.app.shared.utils.dependencies import get_settings

load_dotenv()

settings = get_settings()
DATASET_NAME = "eval_questions&answers_EN"
API_ENDPOINT = "chat_agent_endpoint"  # Replace with actual endpoint
API_KEY = settings.WL_API_KEY

llm = ChatMistralAI(model_name="mistral-small-latest")
ls_client = Client()
mistral_client = Mistral(api_key=settings.MISTRAL_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful assistant specializing in sustainability "
    "and the UN Sustainable Development Goals."
    "You are a helpful assistant specializing in sustainability "
    "and the UN Sustainable Development Goals. Your answer needs "
    "to be justified by resources and text contents from these "
    "resources that have been used to generate the answer MUST "
    "be explicitely cited."
)


class AnswerWithDocuments(TypedDict):
    '''An answer to the user question along with source documents for the answer.'''

    answer: str
    documents: Annotated[
        list[str] | None, None, "A list of source documents for the answer."
    ]


structured_llm = llm.with_structured_output(AnswerWithDocuments)


@traceable
def get_wl_answer(question: str) -> dict:
    try:
        with httpx.Client(timeout=120) as http:
            resp = http.post(
                API_ENDPOINT,
                json={"query": question, "sdg-filter": [], "corpora": [], "thread-id": str(uuid.uuid4())},
                headers={"x-API-Key": API_KEY, "origin": ""},
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "answer": data.get("content") or "",
            "documents": data.get("docs") or [],
        }
    except Exception as exc:
        print(f"Error in get_wl_answer: {exc}")
        return {"answer": f"[ERROR] {exc}", "documents": []}


@traceable
def get_llm_answer(question: str) -> dict:
    try:
        resp = structured_llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
        return resp
    except Exception as exc:
        return {"answer": f"[ERROR] {exc}", "documents": []}


existing = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
dataset = existing[0]


# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


# Grader LLM
grader_llm = ChatMistralAI(model_name="mistral-small-latest", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['output']}
STUDENT ANSWER: {outputs['answer']}"""
    # Run evaluator
    grade = grader_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )
    return grade["correct"]


# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


# Grade prompt
relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = ChatMistralAI(model_name="mistral-small-latest", temperature=0).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)


# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]


# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = ChatMistralAI(model_name="mistral-small-latest", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)


# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc['payload']['slice_content'] for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": answer}
    ])
    return grade["grounded"]


# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatMistralAI(model_name="mistral-small-latest", temperature=0).with_structured_output(
    RetrievalRelevanceGrade, method="json_schema", strict=True
)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc['payload']['slice_content'] for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]


def wl_target(inputs: dict) -> dict:
    return get_wl_answer(inputs["question"])


def llm_target(inputs: dict) -> dict:
    return get_llm_answer(inputs["question"])


def run_evaluation(use_wl: bool = False):
    if use_wl:
        print("Running evaluation with WeLearn API as target...")
        ls_client.evaluate(
            wl_target,
            data=DATASET_NAME,
            evaluators=[correctness, relevance, groundedness, retrieval_relevance],
            experiment_prefix="RAG_metrics_WL",
            metadata={"version": "mistral-small-latest"},
        )
    else:
        print("Running evaluation with raw LLM output as target...")
        ls_client.evaluate(
            llm_target,
            data=DATASET_NAME,
            evaluators=[correctness, relevance],
            experiment_prefix="RAG_metrics_LLM",
            metadata={"version": "mistral-small-latest"},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangSmith eval: /chat/agent (with retrieval) vs raw LLM"
    )
    parser.add_argument(
        "--use-wl",
        action="store_true",
        help="use WeLearn API (default: False), if not set will use raw LLM output as target for evaluation",
    )
    args = parser.parse_args()

    run_evaluation(use_wl=args.use_wl)
