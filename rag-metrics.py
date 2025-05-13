import argparse
import asyncio
import csv
import logging
import random
import time
from itertools import chain
from statistics import mean
from typing import Dict, List

import requests  # type: ignore
from datasets import Dataset  # type: ignore
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from prettytable import PrettyTable
from ragas import evaluate  # type: ignore
from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
from ragas.llms import LangchainLLMWrapper  # type: ignore
from ragas.metrics import (  # type: ignore
    AspectCritic,
    ContextRelevance,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseGroundedness,
    ResponseRelevancy,
)

from src.app.api.dependencies import get_settings
from src.app.services.abst_chat import AbstractChat

logger = logging.getLogger(__name__)

load_dotenv()

random.seed(42)  # nosec B311

with open("eval_questions&answers.csv") as f:
    q_and_a = [row for row in csv.DictReader(f)]

# init chat client
settings = get_settings()

chat = AbstractChat(
    model="azure/gpt-4o-mini",
    API_KEY=settings.AZURE_API_KEY,
    API_BASE=settings.AZURE_API_BASE,
    API_VERSION=settings.AZURE_API_VERSION,
)


# init test chat client
model = "gpt-4o-mini"

evaluator_llm = LangchainLLMWrapper(
    AzureChatOpenAI(
        api_version=settings.AZURE_API_VERSION,
        api_key=settings.AZURE_API_KEY,
        azure_endpoint=settings.AZURE_API_BASE,
        azure_deployment=model,
        model=model,
        validate_base_url=False,
    )
)

models = {
    "fr": "dangvantuan/sentence-camembert-base",
    "en": "sentence-transformers/all-MiniLM-L6-v2",
}


class ObjToClass:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def calculate_mean(eval_results, key):
    """
    Helper function to calculate the rounded mean of a specific key in eval_results.
    Handles both flat and nested lists.
    """

    values = [
        item
        for sublist in [v[key] for _, v in eval_results.items()]
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    return round(mean(values), 3)


async def get_messages(
    all_corpus: bool = False, reranking: bool = False, vanilla: bool = False
):
    print("Getting messages")
    sample_q_and_a: List[Dict[str, str]] = random.sample(
        q_and_a, 12
    )  # This is where the number of questions used for the evaluation can be modified (default value: 12 i.e. 10% of the total number of questions in the dataset)
    relevance = 0.75 if reranking else 1

    if vanilla:
        corpus = [
            {
                "corpus": "no_corpus_en_all-minilm-l6-v2",
                "name": "No Resources",
                "lang": "en",
                "model": "all-minilm-l6-v2",
            },
            {
                "corpus": "no_corpus_fr_sentence-camembert-base",
                "name": "No Resources",
                "lang": "fr",
                "model": "sentence-camembert-base",
            },
        ]
    else:

        resp = requests.get(
            url="https://api.welearn.k8s.lp-i.dev/api/v1/search/collections",
            headers={"x-API-Key": "welearn"},
        )
        corpus = resp.json()

        if all_corpus:
            names: Dict[str, List[str]] = {"en": [], "fr": []}
            for corp in corpus:
                names[corp["lang"]].append(corp["name"])

            corpus = [
                {
                    "corpus": "all_corpus_en_all-minilm-l6-v2",
                    "name": "|".join(names["en"]),
                    "lang": "en",
                    "model": "all-minilm-l6-v2",
                },
                {
                    "corpus": "all_corpus_fr_sentence-camembert-base",
                    "name": "|".join(names["fr"]),
                    "lang": "fr",
                    "model": "sentence-camembert-base",
                },
            ]

    eval_data = {}
    for corp in corpus:
        logger.info("Processing corpus: {} ({})".format(corp["name"], corp["lang"]))
        corpus_data = []
        # In each corpus we iterate over the same 12 Q&A
        for s in sample_q_and_a:
            for attempt in range(5):
                try:
                    query = s["question_{}".format(corp["lang"])]

                    if corp["name"] == "No Resources":
                        # If there are no resources, we just return the question
                        answer = await chat.chat_message(
                            query=query,
                            history=[],
                            docs=[],
                            subject="General",
                            should_check_lang=False,
                        )
                        corpus_data.append(
                            {
                                "user_input": query,
                                "response": answer,
                                "retrieved_contexts": [],
                            }
                        )
                        break

                    payload = {
                        "sdg_filter": [i for i in range(1, 18)],
                        "query": query,
                        "nb_results": 15,
                        "relevance_factor": relevance,
                        "corpora": corp["name"].split("|"),
                    }

                    resp = requests.post(
                        "https://api.welearn.k8s.lp-i.dev/api/v1/search/by_document",
                        headers={"x-API-Key": "welearn"},
                        json=payload,
                    )

                    resp_list = [
                        ObjToClass(
                            {
                                "score": doc["score"],
                                "payload": ObjToClass(doc["payload"]),
                            }
                        )
                        for doc in resp.json()
                    ]

                    context = [doc.payload.slice_content for doc in resp_list]

                    answer = await chat.chat_message(
                        query=query,
                        history=[],
                        docs=resp_list,
                        subject="General",
                        should_check_lang=False,
                    )

                    corpus_data.append(
                        {
                            "user_input": query,
                            "response": answer,
                            "retrieved_contexts": context,
                        }
                    )

                    break
                except Exception as e:
                    logger.info(
                        'Attempt {}/5 to get process question "{}" failed'.format(
                            str(attempt + 1), s["question_{}".format(corp["lang"])]
                        )
                    )
                    logger.info(e)
            else:
                logger.info(
                    "Failed to process question: {}".format(
                        s["question_{}".format(corp["lang"])]
                    )
                )
                continue
        eval_data[(corp["name"], corp["lang"])] = corpus_data

    return eval_data, sample_q_and_a


def get_results(eval_data, sample_q_and_a):
    print("Getting results")
    eval_results = {}
    for k, v in eval_data.items():
        logger.info("Evaluating results for {} ({})".format(k[0], k[1]))
        langchain_hf_embeddings = HuggingFaceEmbeddings(
            model_name=models[k[1]], model_kwargs={"device": "cpu"}
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(langchain_hf_embeddings)
        result = evaluate(
            dataset=Dataset.from_list(v),
            metrics=[
                LLMContextPrecisionWithoutReference(),
                ResponseRelevancy(),
                Faithfulness(),
                ContextRelevance(),
                ResponseGroundedness(),
            ],
            llm=evaluator_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
        )
        eval_results[(k[0], k[1])] = result
    return eval_results, sample_q_and_a


def print_results(eval_results, sample_q_and_a):
    print("Printing results")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open("eval_results_{}.csv".format(timestr), "w") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "corpus",
                "lang",
                "context_precision",
                "answer_relevancy",
                "faithfulness",
                "nv_context_relevance",
                "nv_response_groundedness",
            ],
        )
        writer.writeheader()
        for k, v in eval_results.items():
            writer.writerow(
                {
                    "corpus": k[0],
                    "lang": k[1],
                    "context_precision": v["llm_context_precision_without_reference"],
                    "answer_relevancy": v["answer_relevancy"],
                    "faithfulness": v["faithfulness"],
                    "nv_context_relevance": v["nv_context_relevance"],
                    "nv_response_groundedness": v["nv_response_groundedness"],
                }
            )

    print("\nQUESTIONS USED FOR EVALUATION")
    for s in sample_q_and_a:
        print("-", s["question_fr"])

    print("\nRESULTS")
    table = PrettyTable()
    table.field_names = [
        "Context Precision",
        "Response Relevancy",
        "Faithfulness",
        "Context Relevance",
        "Response Groundedness",
    ]

    table.add_row(
        [
            calculate_mean(eval_results, "llm_context_precision_without_reference"),
            calculate_mean(eval_results, "answer_relevancy"),
            calculate_mean(eval_results, "faithfulness"),
            calculate_mean(eval_results, "nv_context_relevance"),
            calculate_mean(eval_results, "nv_response_groundedness"),
        ]
    )
    table.align = "r"
    print(table)

    print("\nRESULTS BY CORPUS")
    table_corpus = PrettyTable()
    table_corpus.field_names = [
        "Corpus",
        "Context Precision",
        "Response Relevancy",
        "Faithfulness",
        "Context Relevance",
        "Response Groundedness",
    ]
    for corpus in set([k[0] for k in eval_results.keys()]):
        table_corpus.add_row(
            [
                corpus,
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[0] == corpus},
                    "llm_context_precision_without_reference",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[0] == corpus},
                    "answer_relevancy",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[0] == corpus},
                    "faithfulness",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[0] == corpus},
                    "nv_context_relevance",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[0] == corpus},
                    "nv_response_groundedness",
                ),
            ]
        )
    table_corpus.sortby = "Corpus"
    table_corpus.align = "r"
    print(table_corpus)

    print("\nRESULTS BY LANGUAGE")
    table_lang = PrettyTable()
    table_lang.field_names = [
        "Language",
        "Context Precision",
        "Response Relevancy",
        "Faithfulness",
        "Context Relevance",
        "Response Groundedness",
    ]
    for lang in set([k[1] for k in eval_results.keys()]):
        table_lang.add_row(
            [
                lang,
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[1] == lang},
                    "llm_context_precision_without_reference",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[1] == lang},
                    "answer_relevancy",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[1] == lang},
                    "faithfulness",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[1] == lang},
                    "nv_context_relevance",
                ),
                calculate_mean(
                    {k: v for k, v in eval_results.items() if k[1] == lang},
                    "nv_response_groundedness",
                ),
            ]
        )
    table_lang.sortby = "Language"
    table_lang.align = "r"
    print(table_lang)

    print("\nDETAILED RESULTS")
    table_details = PrettyTable()
    table_details.field_names = [
        "Corpus",
        "Language",
        "Context Precision",
        "Response Relevancy",
        "Faithfulness",
        "Context Relevance",
        "Response Groundedness",
    ]
    for k, v in eval_results.items():
        table_details.add_row(
            [
                k[0],
                k[1],
                round(
                    mean(v["llm_context_precision_without_reference"]),
                    3,
                ),
                round(
                    mean(v["answer_relevancy"]),
                    3,
                ),
                round(
                    mean(v["faithfulness"]),
                    3,
                ),
                round(
                    mean(v["nv_context_relevance"]),
                    3,
                ),
                round(
                    mean(v["nv_response_groundedness"]),
                    3,
                ),
            ]
        )
    table_details.sortby = "Corpus"
    table_details.align = "r"
    print(table_details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script computes RAG metrics for a set of questions and answers"
    )
    parser.add_argument(
        "--all_corpus", action="store_true", help="use all corpus to get messages"
    )
    parser.add_argument("--reranking", action="store_true", help="use reranking")
    parser.add_argument(
        "--vanilla", action="store_true", help="no use of WeLearn resources"
    )
    args = parser.parse_args()

    mess, q_and_a = asyncio.run(
        get_messages(
            all_corpus=args.all_corpus, reranking=args.reranking, vanilla=args.vanilla
        )
    )
    print("Messages received")
    res, q_and_a = get_results(mess, q_and_a)
    print_results(res, q_and_a)
