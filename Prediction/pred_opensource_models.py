import os
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
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
    parser.add_argument('--model_name', type=str, default=None, help="raw model name for evaluation", choices=["rwkv-4-14b-pile","long_llama_3b","LLaMA-2-7B-32K","chatglm2-6b-32k"])
    parser.add_argument('--task', type=str, default=None, help="long context understanding tasks in LooGLE", choices=["shortdep_qa","longdep_qa","longdep_summarization","shortdep_cloze"])
    parser.add_argument('--max_length', type=int, default=None, help="the max length of input prompt")

    parser.add_argument('--model_path', type=str, default="./Models/") 
    parser.add_argument('--output_path', type=str, default="./Output/")

    return parser.parse_args(args)



def get_pred(model, data_instance, tokenizer, max_length, max_gen, prompt_format, device):

    ans, groundtruth = [], []
    preds = {}
    raw_inputs = data_instance['input']
    if data_instance['qa_pairs'] == 'none':
        preds['qa_pairs'] = data_instance['qa_pairs']
        json_obj = {'input': raw_inputs}

        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        
        input_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(device)
        context_length = input_ids.shape[-1]
        with torch.no_grad():
            output = model.generate(input_ids,max_new_tokens=max_gen,temperature=1.0,num_beams=1,do_sample=False,repetition_penalty=float(2))[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        ans.append(pred)
        groundtruth.append(raw_inputs)

    else:
        preds['qa_pairs'] = eval(data_instance['qa_pairs'])
        for j in eval(data_instance['qa_pairs']):

            json_obj = {'Q':j['Q'], 'input': raw_inputs}
                
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
            
            input_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(device)
            context_length = input_ids.shape[-1]
            with torch.no_grad():
                output = model.generate(input_ids,max_new_tokens=max_gen,temperature=1.0,num_beams=1,do_sample=False,repetition_penalty=float(2))[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

            # del output, input_ids
            # torch.cuda.empty_cache()

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path + args.model_name, trust_remote_code=True,torch_dtype=torch.bfloat16 ).to(device)
    model.eval()
    
    task2prompt = json.load(open("./config/task2prompt.json", "r"))
    task2maxlen = json.load(open("./config/task2maxlen.json", "r"))
    prompt_format = task2prompt[args.task]
    max_gen = task2maxlen[args.task]

    for i in data:
        preds = get_pred(model, i, tokenizer, args.max_length, max_gen, prompt_format, device)

        with open(args.output_path + args.task + '_' + args.model_name+".jsonl", "a+") as g:
            g.write(json.dumps(preds)+'\n')
