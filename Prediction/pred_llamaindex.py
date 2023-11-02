import os
import torch
import json
import argparse
from datasets import load_dataset
from llama_index import GPTVectorStoreIndex, Document, ServiceContext
from llama_index.indices.prompt_helper import PromptHelper
from transformers import AutoTokenizer
import openai
import tiktoken
#import GPUtil
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
    parser.add_argument('--model_name', type=str, default="llama-index", help="raw model name for evaluation")
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


def get_pred(data_instance, tokenizer, max_length, max_gen, prompt_format):
    
    ans, groundtruth = [], []
    preds = {}
    raw_inputs = data_instance['input']
    documents = [Document(text=raw_inputs)]
    prompt_helper = PromptHelper(
        context_window=max_length + 1000,
        num_output=max_gen,
        chunk_size_limit=1024,
        chunk_overlap_ratio=0.1,
    )

    service_context = ServiceContext.from_defaults(
        context_window=max_length + 1000,
        num_output=max_gen,
        prompt_helper=prompt_helper,
        chunk_size_limit=1024,
    )
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    if data_instance['qa_pairs'] == 'none':
        preds['qa_pairs'] = data_instance['qa_pairs']
        json_obj = {'input': raw_inputs}

        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer.encode(prompt)
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half])+tokenizer.decode(tokenized_prompt[-half:])
        
        rsp = query_engine.query(prompt).response    
        ans.append(rsp)
        groundtruth.append(data_instance["output"])

    else:
        preds['qa_pairs'] = eval(data_instance['qa_pairs'])
    
        for j in eval(data_instance['qa_pairs']): 
            json_obj = {'Q':j['Q'], 'input': raw_inputs}
            
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer.encode(prompt)
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half])+tokenizer.decode(tokenized_prompt[-half:])
            
            rsp = query_engine.query(prompt).response    
            ans.append(rsp)
            groundtruth.append(j['A'])


    preds['llm_output'] = ans
    preds['output'] = groundtruth  
    return preds


def loads(path, task):
    data = []
    with open(path+task+".jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    # data = load_dataset('bigainlco/LooGLE', args.task, split="test")
    data = loads("LooGLE-testdata/", args.task)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    task2prompt = json.load(open("./config/task2prompt.json", "r"))
    task2maxlen = json.load(open("./config/task2maxlen.json", "r"))
    prompt_format = task2prompt[args.task]
    max_gen = task2maxlen[args.task]

    for i in data:
        predictions = get_pred(i, tokenizer, args.max_length, max_gen, prompt_format)

        with open(args.output_path + args.task + '_' + args.model_name + ".jsonl", "a+") as g:
            g.write(json.dumps(predictions)+'\n')

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               