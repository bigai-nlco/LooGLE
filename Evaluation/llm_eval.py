import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from bert_score import score
import numpy as np
import argparse
import openai, os
from llm_score import (
    get_gpt4_score
)

def evaluation(data, scores, functions, task):
    for i in range(len(data["output"])):
        hyp, ref = data["llm_output"][i], data["output"][i]
        if "qa_pairs" in data and data["qa_pairs"] != "none":
            question = data["qa_pairs"][i]["Q"]
        else:
            question = ""

        for j in functions:
            if j not in scores:
                scores[j] = []
            scores[j].append(eval(j)(question, ref, hyp, task))

    return scores


def get_accuracy(result, functions, task):
    final_score = {}
    for i in functions:
        res = result[i]
        if "qa" in task:
            final_score[i] = res.count("True") / (res.count("True") + res.count("False"))
        else:
            final_score[i] = np.mean(res)
    return final_score



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=None, help="model name for evaluation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="long context understanding tasks in LooGLE",
        choices=[
            "shortdep_qa",
            "longdep_qa",
            "longdep_summarization",
        ],
    )
    parser.add_argument("--output_path", type=str, default="./Output/")
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="llm",
        help="evaluation method for LLM predictions",
        choices=["llm"],
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    openai_api_key = os.environ["OPENAI_API_KEY"]
    eval_functions = ["get_gpt4_score"]

    score_result = {}

    with open(
        args.output_path + args.task + "_" + args.model_name + ".jsonl", "r"
    ) as f:
        for line in f.readlines():
            ds_llm = json.loads(line)
            score_result = evaluation(ds_llm, score_result, eval_functions, args.task)


        print(get_accuracy(score_result, eval_functions, args.task))

