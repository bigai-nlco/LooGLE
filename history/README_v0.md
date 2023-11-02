<div align="center" id="title"> <img src="./figs/LooGle_logo.png" width=256px /> </div>

<h2 align="center">Long Context Generic Language Evaluation benchmark for LLM long context understanding</h2>
<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://www.python.org/downloads/release/python-380/">
        <img alt="Documentation" src="https://img.shields.io/badge/Python-3.8+-blue.svg">
    </a>
</p>

![](figs/overview_page1.png)

**LooGLE** is a comprehensive evaluation benchmark for LLM long context understanding which contains up-to-date  (all after 2022) and extremely long realistic documents (over 24k tokens per document, many of which exceed 100k words) from diverse domains and categories. Details statistics of our dataset can be seen in the table below.

**Short and long dependency tasks  📜**    LooGLE is composed of 7 major tasks with a total of 6k+ questions to evaluate LLMs' ability to understand both short and long dependency content. We refer to ``long dependency" tasks as those that require the understanding of the inter-dependency across multiple shreds of evidence widely spanning over the entire long text. We delicately design 5 types of long dependency tasks, including comprehension_and_reasoning, computation, timeline reorder, multiple information retrieval, and summarization.

Specifically, we recruited a group of human annotators to read 145 long documents in our benchmark and manually create 1k+ long dependency Question-Answer (QA) instances, despite the high costs and huge effort involved in this process. These 1k+ high-quality QA pairs are each cross-validated 3 times by 2 annotators, aiming to provide the currently most accurate evaluation of LLMs’ ability on long dependency questions.

**Long context evaluation  📊**  In order to provide more comprehensive and general results, LooGLE not only relies on automatic metrics based on n-gram matching and semantic similarity commonly used in previous benchmarks. Besides, it leverages GPT4-as-judgment and human evaluation to get an overall performance for reference. In the first version, we conduct the evaluation of 8 representative LLMs on LooGLE as the baselines. We specifically select LLMs which have made great effort in addressing the challenge of understanding long contexts by utilizing flash attention, position interpolation, optimized Transformer and finetuning, external memory etc. We will also keep up with the latest releases of long-context understanding LLMs in our benchmarks.

We hope LooGLE can help researchers and developers track the progress of long-context LLMs and understand the strengths/shortcomings of different methods. 


## 📌 Statistics of LooGLE

![](figs/table.png)

## ✏️ Table of Contents
- [Capability leaderboard](#-capability-leaderboard)
- [Quick Start](#-quick-start)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [Contacts](#-contacts)

## 🚀 Capability leaderboard
The overall performance comparisons of different models on different tasks in our dataset are shown in the figure below.

![](figs/overview_performance.png)


## 💁 Quick Start
### Step 1. Download the data 
You can download and load the **LooGLE** data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/bigainlco/LooGLE)):

```python
from datasets import load_dataset

datasets = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "longdep_summarization"]

for testset in datasets:
    data = load_dataset('!!/LooGLE', testset, split='test')
    # evaluate your model
```

All data in **LooGLE** are standardized to the following format:
```json
{
    "input": "The original long input texts",
    "title": "The title of the given document",  //for arxiv paper, we use "arxiv_id" instead as an identical ID
    "qa_pairs":[
            {
                "Q": "Question to ask based on the given input",
                "A": "Groundtruth answer for the question",
                "S": [ "One or more evidence (complete sentences) for answering the question, which are extracted directly from the original input"
                ]
            },  
        ]        // There are multiple questions and corresponding answers in the list (each of them is in json format)
                 // For arxiv paper summarization, we use "none" instead for non-qa/non-cloze tasks
    "output": "none"   // the predicted outputs of LLM given the long input and instructions, which is initialized as "none"
```
To mention that, in long dependency QA data, we add an extra key `type` for each question in json to indicate the 4 types of long dependency tasks(apart from summarization): comprehension_and_reasoning, computation, timeline reorder, multiple information retrieval. Each task of data can be employed to test the specific capability in a long context.


### Step 2. Generate the prediction results
We test LLMs using python codes under the path `Prediction/`. There are 3 `.py` file for corresponding types of models. We use the command below to by selecting the model you want to evaluate via `--model_name` and the specific task via `--task`:

For GPT-3.5-turbo and GPT4:
```
python Prediction/pred_gpt_models.py  --model_name gpt-3.5-turbo --task shortdep_qa --max_length 500
```

For LlamaIndex:
```
python Prediction/pred_llamaindex.py --task shortdep_qa --max_length 500
```

For other open-source models (take chatglm2-6b-32k as an example):
```
python Prediction/pred_opensource_models.py  --model_name chatglm2-6b-32k --task shortdep_qa --max_length 500
```

Open-source models can be download and loaded from `Models/` by default, you can change the path via `--model_path`

You can also determine the input long texts and output result through  `--rawtext_path` and `--output_path`.  

Please note that in `Prediction/config/`, we provide the prompt format suitable for each task and the maximum generation length. The input parameter `--max_length` limits the max length of input prompt for selcted model. Feel free to modify them to better suit the model you want to evaluate. After modification, the data will be automatically organized according to the new format to get the corresponding model output.

We test all the open-source baselines with a single 80G A800 GPU in BF16 precision. If you encounter the OOM problem, please refer to more multiple GPUs inference techniques. For Llama-2 based models, we recommend using Flash Attention for optimization and saving GPU memory The relevant dependencies can be installed according to the code base of [Flash Attention](https://github.com/Dao-AILab/flash-attention).



## 📊 Evaluation

The outputs of different models under LooGLE can be obtained from `Data/output/` by default. Given the prediction file generated in Step 2, we run the evaluation code in [Evaluation/eval.py](Evaluation/eval.py):

```
python Evaluation/eval.py --model_name chatglm2-6b-32k --task shortdep_qa --eval_metric automatic
```

Besides the parameters specifying the `--model_name` and `--task`, we provide `--eval_metric` for users to choose the method for evaluation from [`automatic_sim`, `llm`, `automatic_match`] . Both the automatic metric and LLM-as-judge can be applied in LooGLE to provide a more comprehensive assessment.

Automatic metrics based on semantic similarity matching including Bleu, Rouge, Meteor, Bertscore and exact/partial match are supported. Feel free to add other metrics for your needs.  

We also employ GPT4 as judgment since it has shown that the GPT4 evaluator exhibits high consistency with human evaluation as a reliable annotator to some extent. The prompt of GPT4 given in the repo can be altered for further evaluation.

## 📝 Citation
If you would like to use our data or find our work interesting, please cite:
```bibtex
@article{li2023loogle,
  title={Can Long-Context Language Models Understand Long Contexts?},
  author={ Li, Jiaqi and Wang, Mengmeng and Zheng, Zilong and Zhang, Muhan },
  url={https://github.com/bigai-nlco/LooGLE}
  year={2023}
}
```

## 📣 Contacts

We sincerely appreciate human annotators for their valuable contributions to creating high-quality long-dependency QA tasks.
We are very pleased to answer any questions about LooGLE: nlp@bigai.ai
