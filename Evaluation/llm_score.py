import json
import numpy as np
import openai


def get_gpt4_score(question, reference, hypothesis, task):
    if "qa" in task:
        p = "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' ."

        prompt = [{"role": "system", "content": p,},
        {
            "role": "user",
            "content": "Question: "
            + question
            + "\n"
            + "groudtruth = "
            + reference
            + "\n"
            + "predict_answer = "
            + hypothesis,
        }]

    else:
        # p = "There is a groundtruth summary of a arxiv paper and a auto-generated summary .Please Compare generated summary with the goundtruth and evaluate the generated summary from the perspectives of information completeness, consistency, fluency, and grammar by giving a score within the range of 0 to 100."
        prompt_format = "There is a groundtruth summary of a arxiv paper and a auto-generated summary .Please Compare generated summary with the goundtruth and evaluate the generated summary from the perspectives of information completeness, consistency, fluency, and grammar by giving a score within the range of 0 to 100. \nGroundtruth = {} \nGenerated = {} \nScore = "
        prompt = prompt_format.format(reference, hypothesis)
        prompt = [{"role": "system", "content": prompt}]
        
    rr = openai.ChatCompletion.create(
        model="gpt-4",
        messages=prompt,
        temperature=0.0,
        top_p=1,
        max_tokens=10,
        frequency_penalty=0,
        presence_penalty=0,
    )
    rsp = rr["choices"][0]["message"]["content"]

    if "qa" in task:
        return rsp
    else:
        return int(rsp)

