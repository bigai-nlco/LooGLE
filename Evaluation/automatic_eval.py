import json
import openai
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from bert_score import score
import numpy as np
import argparse
import openai
from automatic_metrics import (
    get_bleu_score,
    get_rouge_score,
    get_meteor_score,
    get_bertscore,
    get_exact_match,
    get_partial_match
)


def evaluation(data, scores, functions, task):
    for i in range(len(data["output"])):
        hyp, ref = data["llm_output"][i], data["output"][i]
        if hyp == '':
            hyp = 'None'
        if "qa_pairs" in data:
            if data["qa_pairs"] != "none":
                question = data["qa_pairs"][i]["Q"]
            else:
                question = ""

        for j in functions:
            if j not in scores:
                scores[j] = []
            scores[j].append(eval(j)(question, ref, hyp, task))

    return scores


def get_semantic_matching(result, functions):
    final_score = {}
    for i in functions:
        if type(result[i][0]) is tuple:
            l = result[i]
            final_score[i] = [np.mean([i[j] for i in l]) for j in range(len(l[0]))]
        else:
            final_score[i] = np.mean(result[i])
    return final_score


def get_match_score(result, functions):
    final_score = {}
    for i in functions:
        match_count = np.sum([j[0] for j in result[i]])
        all_count = np.sum([j[1] for j in result[i]])
        final_score[i] = round(match_count / all_count, 4)
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
            "shortdep_cloze",
            "longdep_qa",
            "longdep_summarization",
        ],
    )
    parser.add_argument("--output_path", type=str, default="./Output/")
    parser.add_argument(
        "--eval_metric",
        type=str,
        default=None,
        help="evaluation method for LLM predictions",
        choices=["automatic_sim", "automatic_match"],
    )

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    if args.eval_metric == "automatic_sim":
        eval_functions = [
            "get_bleu_score",
            "get_rouge_score",
            "get_meteor_score",
            "get_bertscore"
        ]
    elif args.eval_metric == "automatic_match":
        eval_functions = ["get_exact_match", "get_partial_match"]

    score_result = {}
    with open(
        args.output_path + args.task + "_" + args.model_name + ".jsonl", "r"
    ) as f:
        for line in f.readlines():
            ds_llm = json.loads(line)
            score_result = evaluation(ds_llm, score_result, eval_functions, args.task)


        if args.eval_metric == "automatic_sim":
            print(get_semantic_matching(score_result, eval_functions))
        elif args.eval_metric == "automatic_match":
            print(get_match_score(score_result, eval_functions))

