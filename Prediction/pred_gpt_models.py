import os
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from datasets import load_dataset
import tiktoken
# import GPUtil
stopped_num = 10000000    
delay = 10  
# Gpus = GPUtil.getGPUs()

def get_gpu_info():
    gpulist = []
    GPUtil.showUtilization()

    for gpu in Gpus:
        print('gpu.id:', gpu.id)
        print('total GPU:', gpu.memoryTotal)
        print('GPU usage：', gpu.memoryUsed)
        print('gpu usage percent:', gpu.memoryUtil * 100)
        gpulist.append([ gpu.id, gpu.memoryTotal, gpu.memoryUsed,gpu.memoryUtil * 100])

    return gpulist


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help="raw model name for evaluation", choices=["gpt-3.5-turbo-16k", "gpt-4"])
    parser.add_argument('--task', type=str, default=None, help="long context understanding tasks in LooGLE", choices=["shortdep_qa","longdep_qa","longdep_summarization","shortdep_cloze"])
    parser.add_argument('--max_length', type=int, default=None, help="the max length of input prompt")
 
    parser.add_argument('--model_path', type=str, default="./Models/") 
    parser.add_argument('--output_path', type=str, default="./Output/")

    return parser.parse_args(args)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_pred(model, data_instance, tokenizer, max_length, max_gen, prompt_format):
    
    ans, groundtruth = [], []
    preds = {}
    raw_inputs = data_instance['input']
    if data_instance['qa_pairs'] == 'none':
        preds['qa_pairs'] = data_instance['qa_pairs']
        json_obj = {'input': raw_inputs}

        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer.encode(prompt)
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half]) + tokenizer.decode(tokenized_prompt[-half:])

        rsp = openai.ChatCompletion.create(
            model = model,
            messages = [{"role": "system", "content":prompt}],
            temperature = 0.0,
            top_p = 1,
            max_tokens = max_gen,
            frequency_penalty = 0,
            presence_penalty = 0
                )
        pred = rsp['choices'][0]['message']['content']

        ans.append(pred)
        groundtruth.append(raw_inputs)

    else:
        preds['qa_pairs'] = eval(data_instance['qa_pairs'])
    
        for j in eval(data_instance['qa_pairs']):

            json_obj = {'Q':j['Q'], 'input': raw_inputs}
            
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer.encode(prompt)
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half])+tokenizer.decode(tokenized_prompt[-half:])

            rsp = openai.ChatCompletion.create(
                model = model,
                messages = [{"role": "system", "content":prompt}],
                temperature = 0.0,
                top_p = 1,
                max_tokens = max_gen,
                frequency_penalty = 0,
                presence_penalty = 0
                    )
            pred = rsp['choices'][0]['message']['content']
            ans.append(pred)
            groundtruth.append(j['A'])

    preds['llm_output'] = ans
    preds['output'] = groundtruth

    return preds

# def loads(path, task):
#     data = []
#     with open(path+task+".jsonl", "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             data.append(json.loads(line))
#     return data

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    data = load_dataset('bigainlco/LooGLE', args.task, split="test")
    #data = loads("LooGLE-testdata/", args.task)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    task2prompt = json.load(open("./config/task2prompt.json", "r"))
    task2maxlen = json.load(open("./config/task2maxlen.json", "r"))
    prompt_format = task2prompt[args.task]
    max_gen = task2maxlen[args.task]
    for i in data:
        predictions = get_pred(args.model_name, i, tokenizer, args.max_length, max_gen, prompt_format)
        with open(args.output_path + args.task + '_' + args.model_name+".jsonl", "a+") as g:
            g.write(json.dumps(predictions)+'\n')

