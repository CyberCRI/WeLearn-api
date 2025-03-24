import argparse
import asyncio
import csv
import logging
import random
import time
from statistics import mean
from typing import Dict, List

import requests
from datasets import Dataset  # type: ignore
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from prettytable import PrettyTable
from ragas import evaluate  # type: ignore
from ragas.metrics import answer_correctness, answer_relevancy, answer_similarity, context_precision, context_recall, context_utilization, faithfulness  # type: ignore

from src.app.api.dependencies import get_settings
from src.app.services.abst_chat import ChatFactory

logger = logging.getLogger(__name__)

load_dotenv()

random.seed(42)  # nosec B311

with open("eval_questions&answers.csv") as f:
    q_and_a = [row for row in csv.DictReader(f)]

# init chat client
settings = get_settings()

chat = ChatFactory().create_chat("openai")

chat.init_client()

# init test chat client
model = "gpt-35-turbo-16k"

test_chat = AzureChatOpenAI(
    api_version=settings.API_VERSION,
    api_key=settings.API_KEY,
    azure_endpoint=settings.API_BASE,
    azure_deployment=model,
    model=model,
    validate_base_url=False,
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


async def get_messages(all_corpus: bool = False, reranking: bool = False):
    print("Getting messages")
    sample_q_and_a: List[Dict[str, str]] = random.sample(
        q_and_a, 12
    )  # This is where the number of questions used for the evaluation can be modified (default value: 12 i.e. 10% of the total number of questions in the dataset)
    relevance = 0.75 if reranking else 1

    resp = requests.get("https://api.welearn.k8s.lp-i.dev/api/v1/search/collections")
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
                    ground_truth = s["answer_{}".format(corp["lang"])]
                    
                    payload = {
                        "sdg_filter": [
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
                            ],
                        "query": query,
                        "nb_results": 15,
                        "relevance_factor": relevance,
                        "corpora": corp['name'].split("|"),
                    }
                    resp = requests.post(
                        "https://api.welearn.k8s.lp-i.dev/api/v1/search/by_document",
                        json=payload,
                    )

                    resp_list = [ObjToClass(doc) for doc in resp.json()]

                    context = [
                        doc["payload"]["slice_content"]
                        for (_, doc) in enumerate(resp.json())
                    ]

                    answer = await chat.chat_message(
                        query=query,
                        history=[],
                        docs=resp_list,
                        subject="General",
                        should_check_lang=False,
                    )

                    corpus_data.append(
                        {
                            "question": query,
                            "answer": answer,
                            "contexts": context,
                            "ground_truth": ground_truth,
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
        result = evaluate(
            dataset=Dataset.from_list(v),
            metrics=[
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision,
                answer_similarity,
                answer_correctness,
                context_utilization,
            ],
            llm=test_chat,
            embeddings=SentenceTransformerEmbeddings(
                model_name=models[k[1]], model_kwargs={"device": "cpu"}
            ),
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
                "answer_relevancy",
                "faithfulness",
                "context_recall",
                "context_precision",
                "answer_similarity",
                "answer_correctness",
                "context_utilization",
            ],
        )
        writer.writeheader()
        for k, v in eval_results.items():
            writer.writerow(
                {
                    "corpus": k[0],
                    "lang": k[1],
                    "answer_relevancy": v["answer_relevancy"],
                    "faithfulness": v["faithfulness"],
                    "context_recall": v["context_recall"],
                    "context_precision": v["context_precision"],
                    "answer_similarity": v["answer_similarity"],
                    "answer_correctness": v["answer_correctness"],
                    "context_utilization": v["context_utilization"],
                }
            )

    print("\nQUESTIONS USED FOR EVALUATION")
    for s in sample_q_and_a:
        print("-", s["question_fr"])

    print("\nRESULTS")
    table = PrettyTable()
    table.field_names = [
        "Answer Relevancy",
        "Faithfulness",
        "Context Recall",
        "Context Precision",
        "Answer Similarity",
        "Answer Correctness",
        "Context Utilization",
    ]
    table.add_row(
        [
            round(mean([v["answer_relevancy"] for k, v in eval_results.items()]), 3),
            round(mean([v["faithfulness"] for k, v in eval_results.items()]), 3),
            round(mean([v["context_recall"] for k, v in eval_results.items()]), 3),
            round(mean([v["context_precision"] for k, v in eval_results.items()]), 3),
            round(mean([v["answer_similarity"] for k, v in eval_results.items()]), 3),
            round(mean([v["answer_correctness"] for k, v in eval_results.items()]), 3),
            round(mean([v["context_utilization"] for k, v in eval_results.items()]), 3),
        ]
    )
    table.align = "r"
    print(table)

    print("\nRESULTS BY CORPUS")
    table_corpus = PrettyTable()
    table_corpus.field_names = [
        "Corpus",
        "Answer Relevancy",
        "Faithfulness",
        "Context Recall",
        "Context Precision",
        "Answer Similarity",
        "Answer Correctness",
        "Context Utilization",
    ]
    for corpus in set([k[0] for k in eval_results.keys()]):
        table_corpus.add_row(
            [
                corpus,
                round(
                    mean(
                        [
                            v["answer_relevancy"]
                            for k, v in eval_results.items()
                            if k[0] == corpus
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["faithfulness"]
                            for k, v in eval_results.items()
                            if k[0] == corpus
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["context_recall"]
                            for k, v in eval_results.items()
                            if k[0] == corpus
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["context_precision"]
                            for k, v in eval_results.items()
                            if k[0] == corpus
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["answer_similarity"]
                            for k, v in eval_results.items()
                            if k[0] == corpus
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["answer_correctness"]
                            for k, v in eval_results.items()
                            if k[0] == corpus
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["context_utilization"]
                            for k, v in eval_results.items()
                            if k[0] == corpus
                        ]
                    ),
                    3,
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
        "Answer Relevancy",
        "Faithfulness",
        "Context Recall",
        "Context Precision",
        "Answer Similarity",
        "Answer Correctness",
        "Context Utilization",
    ]
    for lang in set([k[1] for k in eval_results.keys()]):
        table_lang.add_row(
            [
                lang,
                round(
                    mean(
                        [
                            v["answer_relevancy"]
                            for k, v in eval_results.items()
                            if k[1] == lang
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["faithfulness"]
                            for k, v in eval_results.items()
                            if k[1] == lang
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["context_recall"]
                            for k, v in eval_results.items()
                            if k[1] == lang
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["context_precision"]
                            for k, v in eval_results.items()
                            if k[1] == lang
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["answer_similarity"]
                            for k, v in eval_results.items()
                            if k[1] == lang
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["answer_correctness"]
                            for k, v in eval_results.items()
                            if k[1] == lang
                        ]
                    ),
                    3,
                ),
                round(
                    mean(
                        [
                            v["context_utilization"]
                            for k, v in eval_results.items()
                            if k[1] == lang
                        ]
                    ),
                    3,
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
        "Answer Relevancy",
        "Faithfulness",
        "Context Recall",
        "Context Precision",
        "Answer Similarity",
        "Answer Correctness",
        "Context Utilization",
    ]
    for k, v in eval_results.items():
        table_details.add_row(
            [
                k[0],
                k[1],
                round(v["answer_relevancy"], 3),
                round(v["faithfulness"], 3),
                round(v["context_recall"], 3),
                round(v["context_precision"], 3),
                round(v["answer_similarity"], 3),
                round(v["answer_correctness"], 3),
                round(v["context_utilization"], 3),
            ]
        )
    table_details.sortby = "Corpus"
    table_details.align = "r"
    print(table_details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script computes RAG metrics for a set of questions and answers"
    )
    parser.add_argument("--all_corpus", action="store_true", help="use all corpus to get messages") 
    parser.add_argument("--reranking", action="store_true", help="use reranking") 
    args = parser.parse_args()
    
    all_corpus = args.all_corpus
    reranking = args.reranking
    
    mess, q_and_a = asyncio.run(get_messages(all_corpus=all_corpus, reranking=reranking))
    res, q_and_a = get_results(mess, q_and_a)
    print_results(res, q_and_a)
