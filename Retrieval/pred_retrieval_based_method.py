import os
from typing import Any
import torch
import json
import argparse
import openai
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index import GPTVectorStoreIndex, Document, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llms import (
    OpenAI,
    CustomLLM,
    HuggingFaceLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
import tiktoken


class OpenSourceLLM(CustomLLM):
    num_output: int = 0
    model_name: str = ""
    max_length: int = 0
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, num_output, max_length, model_path, model_name) -> None:
        super().__init__()
        self.num_output = num_output
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, model_name), trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_path, model_name), trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        self.model.eval()

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.max_length,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print("input:", prompt)
        input_ids = self.tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids
        input_ids = input_ids.to("cuda")
        context_length = input_ids.shape[-1]
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.num_output,
                temperature=1.0,
                num_beams=1,
                do_sample=False,
                repetition_penalty=float(2),
            )[0]
        text = self.tokenizer.decode(
            output[context_length:], skip_special_tokens=True)

        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama-index",
        help="raw model name for evaluation",
    )
    parser.add_argument(
        "--emb_model_name", type=str, default="", help="embedding_model"
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
            "shortdep_cloze",
        ],
    )
    parser.add_argument(
        "--max_length", type=int, default=None, help="the max length of input prompt"
    )

    parser.add_argument("--model_path", type=str, default="./Models/")
    parser.add_argument("--output_path", type=str, default="./Output/")

    return parser.parse_args(args)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_pred(data_instance, service_context):
    ans, groundtruth = [], []
    preds = {}
    preds["qa_pairs"] = eval(data_instance["qa_pairs"])
    documents = [Document(text=data_instance["input"])]
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    query_engine = index.as_query_engine()

    for j in eval(data_instance["qa_pairs"]):
        rsp = query_engine.query(
            "Question: " + j["Q"] + "\n" + "Answer: ").response
        ans.append(rsp)
        groundtruth.append(j["A"])

    preds["llm_output"] = ans
    preds["output"] = groundtruth
    return preds


def loads(path, task):
    data = []
    with open(path+task+".jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    open_source_model = [
        "rwkv-4-14b-pile",
        "long_llama_3b",
        "LLaMA-2-7B-32K",
        "chatglm2-6b-32k",
    ]
    openai_model = ["gpt-3.5-turbo-16k", "gpt-4"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    task2maxlen = json.load(open("./config/task2maxlen.json", "r"))
    max_gen = task2maxlen[args.task]
    # data = load_dataset("bigainlco/LooGLE", args.task, split="test")
    data = loads("LooGLE-testdata/", args.task)
    if args.model_name in open_source_model:
        llm = OpenSourceLLM(max_gen, args.max_length,
                            args.model_path, args.model_name)
    elif args.model_name in openai_model:
        llm = OpenAI(model=args.model_name)
    else:
        raise NameError("model name not found!")
    embed_model = HuggingFaceEmbeddings(model_name=args.emb_model_name)
    prompt_helper = PromptHelper(
        context_window=args.max_length,
        num_output=max_gen,
        chunk_size_limit=1024,
        chunk_overlap_ratio=0.1,
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        context_window=args.max_length,
        num_output=max_gen,
        embed_model=embed_model,
        prompt_helper=prompt_helper,
        chunk_size_limit=1024,
    )
    for i in data:
        predictions = get_pred(i, service_context)
        with open(
            args.output_path + args.task + "_" + args.model_name + ".jsonl", "a+"
        ) as g:
            g.write(json.dumps(predictions) + "\n")
