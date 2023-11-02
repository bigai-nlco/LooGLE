import json
import numpy as np
import re
import itertools
import argparse
from get_reorder_deviation import (
    location_square_deviation,
    location_mean_deviation,
    swap_deviation,
    swap_distance_deviation
)

from get_max_deviation import (
    get_max_location_square_deviation,
    get_max_location_mean_deviation,
    get_max_swap_deviation,
    get_max_swap_distance_deviation
)

def roman_numerals(text):
	pattern = r"\b[IVXLCDM]+\b"
	return re.findall(pattern, text)


def deduplicate(l):
    new_l=list(set(l))
    new_l.sort(key=l.index)
    return new_l


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=None, help="model name for evaluation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="longdep_qa_reorder",
        help="long context understanding tasks in LooGLE",
        choices=[
            "longdep_qa_reorder"
        ]
    )
    parser.add_argument("--output_path", type=str, default="./Output/")
    

    return parser.parse_args(args)



def evaluation(data, reorder_score, reorder_function):
    for i in range(len(data["output"])):
        hyp, ref = roman_numerals(data['llm_output'][i]), data['output'][i].split(',') 
        #deduplicate(roman_numerals(data['llm_output'][i]))
        #hypothesis.extend(list(set(reference) - set(hypothesis)))

        if hyp == '':
            hyp = []

        for j in reorder_function:
            if j not in reorder_score:
                reorder_score[j] = []

            output = eval(j)(ref, hyp)
            if output != 'None':
                output = eval('get_max_'+j)(len(ref))
            reorder_score[j].append(output)

    return reorder_score


def get_reorder_score(result, functions):
    final_score = {}
    for i in functions:
        res = result[i]
        final_score[i] = np.mean(res)
    return final_score


if __name__ == "__main__":
    args = parse_args()
    eval_functions = ["location_square_deviation","location_mean_deviation","swap_deviation","swap_distance_deviation" ]

    score_result = {}
    cnt = 0
    with open(
        args.output_path + args.task + "_" + args.model_name + ".jsonl", "r") as f:
        for line in f.readlines():
            cnt += 1
            if cnt < 2:
                ds_llm = json.loads(line)
                score_result = evaluation(ds_llm, score_result, eval_functions)


        print(get_reorder_score(score_result, eval_functions))

        
