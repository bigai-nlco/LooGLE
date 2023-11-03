---
layout: default
---

![](https://github.com/bigai-nlco/LooGLE/raw/main/assets/overview_page1.png)

**LooGLE** is a comprehensive evaluation benchmark for LLM long context understanding which contains up-to-date  (all after 2022) and extremely long realistic documents (over 24k tokens per document, many of which exceed 100k words) and 6,000 newly generated questions spanning diverse domains and categories. Details statistics of our dataset can be seen in the table below.

**Short and long dependency tasks  📜**  LooGLE is composed of 7 major tasks to evaluate LLMs' ability to understand both short and long dependency content. We refer to ``long dependency" tasks as those that require the understanding of the inter-dependency across multiple shreds of evidences widely spanning over the entire long text. We delicately design 5 types of long dependency tasks, including comprehension and reasoning, computation, timeline reorder, multiple information retrieval, and summarization.

**Long context evaluation  📊**  In order to provide more comprehensive and general results, LooGLE relies on automatic automatic metrics based on semantic similarity, GPT4-as-judgment and human evaluation to get an overall performance for reference. We conduct the evaluation of 8 representative LLMs. We specifically select LLMs which have made great effort in addressing the challenge of understanding long contexts by utilizing flash attention, position interpolation, optimized Transformer and finetuning, external memory etc. 

LooGLE not only provides a systematic and comprehensive evaluation schema on long-context LLMs, but also sheds light on future development of enhanced models towards “true long-context understanding”.

<!-- <br> -->

## 📌 **Statistics of LooGLE**

<!-- | DataSet | Category | No. Documents | Avg. Words | Task | Subtask | No. Questions |
|:---------:|:----------:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| arXiv papers     |  Physics, Math, <br> Finance, Statistics  <br> Biology, Economics, <br> Computer Science, etc.   |516 | 14,860 | Summarization | - | 516 | -->

![](https://github.com/bigai-nlco/LooGLE/raw/main/assets/table.png)

## ✏️ **Table of Contents**
- [📌 **Statistics of LooGLE**](#-statistics-of-loogle)
- [✏️ **Table of Contents**](#️-table-of-contents)
- [🚀 **Capability leaderboard**](#-capability-leaderboard)
- [💁 **Quick Start**](#-quick-start)
  - [**Step 1. Prerequisites**](#step-1-prerequisites)
  - [**Step 2. Download the data**](#step-2-download-the-data)
  - [**Step 3. Generate the prediction results**](#step-3-generate-the-prediction-results)
  - [**Prediction for retrieval based methods**](#prediction-for-retrieval-based-methods)
- [📊 **Evaluation**](#-evaluation)
  - [**Evaluation on Timeline reorder task**](#evaluation-on-timeline-reorder-task)
- [💡 **Main result on short and long dependency tasks**](#-main-result-on-short-and-long-dependency-tasks)
  - [**Performance of the short dependency tasks**](#performance-of-the-short-dependency-tasks)
  - [**Performance of the long dependency tasks**](#performance-of-the-long-dependency-tasks)
  - [**Impact of input length on long dependency tasks**](#impact-of-input-length-on-long-dependency-tasks)
- [📝 **Citation**](#-citation)
- [📣 **Contacts**](#-contacts)


## 🚀 **Capability leaderboard**
The overall performance comparisons of different models on different tasks in our dataset are shown in the figure below.

![](https://github.com/bigai-nlco/LooGLE/raw/main/assets/overview_performance.png)


## 💁 **Quick Start**
### **Step 1. Prerequisites**
Clone this repo and install the dependencies. The test environment is under torch 2.0.1+cu121.

```bash
cd LooGLE   
conda create -n loogle python=3.9
conda activate loogle
pip install -r requirements.txt
export OPENAI_API_KEY="[your_openai_api_key]"
```


### **Step 2. Download the data** 
You can download and load the **LooGLE** data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/bigainlco/LooGLE)):

```python
from datasets import load_dataset

datasets = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "longdep_summarization"]

for testset in datasets:
    data = load_dataset('bigainlco/LooGLE', testset, split='test')
    # evaluate your model
```
You can also access our sample data [LooGLE-testdata/](LooGLE-testdata/).

All data in **LooGLE** are standardized to the following format:
```json
{
    "input": "The original long input texts",
    "title": "The title of the given document",  //for arxiv paper, we use "title" to refer the identical ID for specific paper
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
To mention that, in long dependency QA data, we add an extra key `type` for each question in json to indicate the 4 types of long dependency tasks(apart from summarization).


### **Step 3. Generate the prediction results**
We test LLMs using 3 python codes under the path [Prediction/](Prediction/) for corresponding types of models. We select the model for evaluation via `--model_name` and the specific task via `--task`. Let's take short dependency QA as an example:

For GPT-3.5-turbo and GPT4:
```
python Prediction/pred_gpt_models.py  --model_name gpt-3.5-turbo-16k --task shortdep_qa --max_length 500
```

For LlamaIndex:
```
python Prediction/pred_llamaindex.py --task shortdep_qa --max_length 500
```

For other open-source models (take chatglm2-6b-32k as an example):
```
python Prediction/pred_opensource_models.py  --model_name chatglm2-6b-32k --task shortdep_qa --max_length 500
```

Open-source models can be download and loaded from [Models/](Models/) by default, you can change the path via `--model_path`

You can also determine the long texts output result through `--output_path`.  

Please note that in `config/`, we provide the prompt format suitable for each task and the maximum generation length. The input parameter `--max_length` limits the max length of input prompt for selcted model. Feel free to modify them to better suit the model you want to evaluate. 

We test all the open-source baselines with a single 80G A800 GPU in BF16 precision. For Llama-2 based models, we recommend using [Flash Attention](https://github.com/Dao-AILab/flash-attention) for optimization and saving GPU memory.


### **Prediction for retrieval based methods**

To evaluate the effectiveness of retrieval techniques for long-context dependency questions, we undertook an extensive experiments by replacing the base LLM model in LlamaIndex with different baseline LLMs. 

For retrieval based methods (take chatglm2-6b-32k as an example):
```
python Retrieval/pred_retrieval_based_method.py --model_name chatglm2-6b-32k --task shortdep_qa --max_length 500 --emb_model_name sentence-transformers/all-mpnet-base-v2
```
Use `--emb_model_name` to set embedding models for retrieval based methods. Here we used all-mpnet-base-v2 as default.


## 📊 **Evaluation**

Given the prediction file generated in Step 2, we run the evaluation code in [Evaluation/](Evaluation/).

For automatic evaluation in short and long dependency QA, summarization task  (eg. short dependency QA):

```
python Evaluation/automatic_eval.py --model_name chatglm2-6b-32k --task shortdep_qa --eval_metric automatic_sim
```

For automatic evaluation in cloze task:

```
python Evaluation/automatic_eval.py --model_name chatglm2-6b-32k --task shortdshortdep_cloze --eval_metric automatic_match
```

For  LLM-as-judge in short and long dependency QA, summarization task (eg. short dependency QA):

```
python Evaluation/llm_eval.py --model_name chatglm2-6b-32k --task shortdep_qa
```

Besides the parameters specifying the `--model_name` and `--task`, we provide `--eval_metric` for users to choose the method for automic evaluation from [`automatic_sim`, `automatic_match`]. 

Automatic metrics based on semantic similarity matching including Bleu, Rouge, Meteor, Bertscore and exact/partial match are supported. Feel free to add other metrics for your needs in  [Evaluation/automatic_metrics.py](Evaluation/automatic_metrics.py). Besides, the prompt of GPT4 given in the repo can be altered for further evaluation.


### **Evaluation on Timeline reorder task**
 We provide four metrics: LSD (location square deviation), LMD (location mean deviation), SD
(swap deviation), and SDD (swap distance deviation) to measure the similarity of numeric sequences for time reorder task with regularized outputs. Details of the implementations can be seen in our paper.

For LLM in long dependency timeline reorder task:
```
python Reorder/automatic_eval.py --model_name chatglm2-6b-32k
```


## 💡 **Main result on short and long dependency tasks**

### **Performance of the short dependency tasks**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax" rowspan="2">Models </th>
    <th class="tg-0lax" rowspan="2">Context </th>
    <th class="tg-0lax" colspan="8">Short dependency QA</th>
    <th class="tg-0lax" colspan="2">Cloze</th>
  </tr>
  <tr>
    <th class="tg-0lax">Bleu1</th>
    <th class="tg-0lax"> Bleu4 </th>
    <th class="tg-0lax">Rouge1 </th>
    <th class="tg-0lax">Rouge4 </th>
    <th class="tg-0lax">RougeL </th>
    <th class="tg-0lax">Meteor score </th>
    <th class="tg-0lax">Bert score </th>
    <th class="tg-0lax">GPT4 score </th>
    <th class="tg-0lax">Exact Match </th>
    <th class="tg-0lax">Partial Match</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">24.61</td>
    <td class="tg-0lax">11.14</td>
    <td class="tg-0lax">61.80</td>
    <td class="tg-0lax">50.73</td>
    <td class="tg-0lax">60.75</td>
    <td class="tg-0lax">32.94</td>
    <td class="tg-0lax">78.72</td>
    <td class="tg-0lax"><b>71.52</b></td>
    <td class="tg-0lax"><b>70.50</b></td>
    <td class="tg-0lax"><b>80.81</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-8k</td>
    <td class="tg-0lax">8K</td>
    <td class="tg-0lax">27.35</td>
    <td class="tg-0lax">14.38</td>
    <td class="tg-0lax"><b>67.59</b></td>
    <td class="tg-0lax"><b>56.01</b></td>
    <td class="tg-0lax"><b>65.77</b></td>
    <td class="tg-0lax"><b>38.56</b></td>
    <td class="tg-0lax"><b>87.93</b></td>
    <td class="tg-0lax">53.99</td>
    <td class="tg-0lax">66.03</td>
    <td class="tg-0lax">76.62</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT3.5-turbo-16k</td>
    <td class="tg-0lax">16K</td>
    <td class="tg-0lax">22.67</td>
    <td class="tg-0lax">9.62</td>
    <td class="tg-0lax">62.56</td>
    <td class="tg-0lax">48.63</td>
    <td class="tg-0lax">60.66</td>
    <td class="tg-0lax">32.58</td>
    <td class="tg-0lax">87.04</td>
    <td class="tg-0lax">66.82</td>
    <td class="tg-0lax">54.64</td>
    <td class="tg-0lax">63.42</td>
  </tr>
  <tr>
    <td class="tg-0lax">LlamaIndex</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax"><b>33.37</b></td>
    <td class="tg-0lax"><b>21.43</b></td>
    <td class="tg-0lax">58.82</td>
    <td class="tg-0lax">42.93</td>
    <td class="tg-0lax">57.08</td>
    <td class="tg-0lax">37.17</td>
    <td class="tg-0lax">86.58</td>
    <td class="tg-0lax">59.61</td>
    <td class="tg-0lax">58.95</td>
    <td class="tg-0lax">66.86</td>
  </tr>
  <tr>
    <td class="tg-0lax">ChatGLM2-6B</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">14.29</td>
    <td class="tg-0lax">6.07</td>
    <td class="tg-0lax">20.50</td>
    <td class="tg-0lax">13.16</td>
    <td class="tg-0lax">20.36</td>
    <td class="tg-0lax">13.08</td>
    <td class="tg-0lax">87.28</td>
    <td class="tg-0lax">23.65</td>
    <td class="tg-0lax">0.05</td>
    <td class="tg-0lax">0.98</td>
  </tr>
  <tr>
    <td class="tg-0lax">LongLLaMa-3B</td>
    <td class="tg-0lax">256k</td>
    <td class="tg-0lax">1.37</td>
    <td class="tg-0lax">0.26</td>
    <td class="tg-0lax">26.97</td>
    <td class="tg-0lax">11.02</td>
    <td class="tg-0lax">26.10</td>
    <td class="tg-0lax">11.34</td>
    <td class="tg-0lax">71.65</td>
    <td class="tg-0lax">13.75</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">2.13</td>
  </tr>
  <tr>
    <td class="tg-0lax">RWKV-4-14B-pile</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax">0.80</td>
    <td class="tg-0lax">0.04</td>
    <td class="tg-0lax">21.70</td>
    <td class="tg-0lax">6.39</td>
    <td class="tg-0lax">20.64</td>
    <td class="tg-0lax">9.41</td>
    <td class="tg-0lax">70.42</td>
    <td class="tg-0lax">8.93</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">LLaMA2-7B-32K</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">0.18</td>
    <td class="tg-0lax">7.25*e-308</td>
    <td class="tg-0lax">1.86</td>
    <td class="tg-0lax">0.00</td>
    <td class="tg-0lax">1.86</td>
    <td class="tg-0lax">1.52</td>
    <td class="tg-0lax">61.53</td>
    <td class="tg-0lax">3.18</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">0.58</td>
  </tr>
</tbody>
</table>
<br>

### **Performance of the long dependency tasks**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">Models </th>
    <th class="tg-0lax">Context </th>
    <th class="tg-0lax">Bleu1</th>
    <th class="tg-0lax"> Bleu4 </th>
    <th class="tg-0lax">Rouge1 </th>
    <th class="tg-0lax">Rouge4 </th>
    <th class="tg-0lax">RougeL </th>
    <th class="tg-0lax">Meteor score </th>
    <th class="tg-0lax">Bert score </th>
    <th class="tg-0lax">GPT4 score </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax" colspan="10">arXiv paper summarization</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">24.50</td>
    <td class="tg-0lax">0.73</td>
    <td class="tg-0lax">27.15</td>
    <td class="tg-0lax">7.10</td>
    <td class="tg-0lax">24.25</td>
    <td class="tg-0lax">19.03</td>
    <td class="tg-0lax">84.04</td>
    <td class="tg-0lax">82.84</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-8k</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax"><b>29.02</b></td>
    <td class="tg-0lax"><b>2.09</b></td>
    <td class="tg-0lax"><b>32.08</b></td>
    <td class="tg-0lax"><b>11.11</b></td>
    <td class="tg-0lax">28.85</td>
    <td class="tg-0lax"><b>22.64</b></td>
    <td class="tg-0lax"><b>84.92</b></td>
    <td class="tg-0lax">85.42</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT3.5-turbo-16k</td>
    <td class="tg-0lax">16k</td>
    <td class="tg-0lax">28.70</td>
    <td class="tg-0lax">1.59</td>
    <td class="tg-0lax">32.04</td>
    <td class="tg-0lax">10.69</td>
    <td class="tg-0lax"><b>28.89</b></td>
    <td class="tg-0lax">22.34</td>
    <td class="tg-0lax">84.82</td>
    <td class="tg-0lax"><b>86.84</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">LlamaIndex</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">22.53</td>
    <td class="tg-0lax">0.63</td>
    <td class="tg-0lax">26.28</td>
    <td class="tg-0lax">6.97</td>
    <td class="tg-0lax">23.73</td>
    <td class="tg-0lax">21.07</td>
    <td class="tg-0lax">83.09</td>
    <td class="tg-0lax">76.35</td>
  </tr>
  <tr>
    <td class="tg-0lax">ChatGLM2-6B</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">0.04</td>
    <td class="tg-0lax">1.60e-310</td>
    <td class="tg-0lax">5.97</td>
    <td class="tg-0lax">8.43E-05</td>
    <td class="tg-0lax">5.82</td>
    <td class="tg-0lax">6.40</td>
    <td class="tg-0lax">73.25</td>
    <td class="tg-0lax">13.23</td>
  </tr>
  <tr>
    <td class="tg-0lax">LongLLaMa-3B</td>
    <td class="tg-0lax">256k</td>
    <td class="tg-0lax">4.24</td>
    <td class="tg-0lax">9.32e-309</td>
    <td class="tg-0lax">4.10</td>
    <td class="tg-0lax">0.52</td>
    <td class="tg-0lax">3.86</td>
    <td class="tg-0lax">3.82</td>
    <td class="tg-0lax">73.41</td>
    <td class="tg-0lax">12.28</td>
  </tr>
  <tr>
    <td class="tg-0lax">RWKV-4-14B-pile</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax">6.28</td>
    <td class="tg-0lax">4.58E-05</td>
    <td class="tg-0lax">6.45</td>
    <td class="tg-0lax">0.74</td>
    <td class="tg-0lax">6.01</td>
    <td class="tg-0lax">6.00</td>
    <td class="tg-0lax">75.28</td>
    <td class="tg-0lax">7.02</td>
  </tr>
  <tr>
    <td class="tg-0lax">LLaMA2-7B-32K</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">0.03</td>
    <td class="tg-0lax">4.66e-310</td>
    <td class="tg-0lax">0.12</td>
    <td class="tg-0lax">0.00</td>
    <td class="tg-0lax">0.12</td>
    <td class="tg-0lax">0.67</td>
    <td class="tg-0lax">71.21</td>
    <td class="tg-0lax">7.60</td>
  </tr>
  <tr>
    <td class="tg-0lax" colspan="10">Long dependency QA</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">8.55</td>
    <td class="tg-0lax">1.40</td>
    <td class="tg-0lax"><b>25.59</b></td>
    <td class="tg-0lax">6.36</td>
    <td class="tg-0lax"><b>24.04</b></td>
    <td class="tg-0lax"><b>11.13</b></td>
    <td class="tg-0lax">80.16</td>
    <td class="tg-0lax"><b>54.09</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-8k</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax"><b>8.94</b></td>
    <td class="tg-0lax">1.01</td>
    <td class="tg-0lax">23.45</td>
    <td class="tg-0lax">6.57</td>
    <td class="tg-0lax">21.69</td>
    <td class="tg-0lax">10.18</td>
    <td class="tg-0lax">85.36</td>
    <td class="tg-0lax">42.12</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT3.5-turbo-16k</td>
    <td class="tg-0lax">16k</td>
    <td class="tg-0lax">6.92</td>
    <td class="tg-0lax"><b>1.81</b></td>
    <td class="tg-0lax">25.02</td>
    <td class="tg-0lax">6.68</td>
    <td class="tg-0lax">23.63</td>
    <td class="tg-0lax">10.40</td>
    <td class="tg-0lax">83.79</td>
    <td class="tg-0lax">45.04</td>
  </tr>
  <tr>
    <td class="tg-0lax">LlamaIndex</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">7.76</td>
    <td class="tg-0lax">1.24</td>
    <td class="tg-0lax">23.62</td>
    <td class="tg-0lax"><b>7.10</b></td>
    <td class="tg-0lax">22.30</td>
    <td class="tg-0lax">10.47</td>
    <td class="tg-0lax">83.87</td>
    <td class="tg-0lax">37.63</td>
  </tr>
  <tr>
    <td class="tg-0lax">ChatGLM2-6B</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">5.55</td>
    <td class="tg-0lax">0.11</td>
    <td class="tg-0lax">9.41</td>
    <td class="tg-0lax">1.93</td>
    <td class="tg-0lax">8.69</td>
    <td class="tg-0lax">4.39</td>
    <td class="tg-0lax"><b>85.78</b></td>
    <td class="tg-0lax">11.50</td>
  </tr>
  <tr>
    <td class="tg-0lax">LongLLaMa-3B</td>
    <td class="tg-0lax">256k</td>
    <td class="tg-0lax">1.04</td>
    <td class="tg-0lax">3.12E-307</td>
    <td class="tg-0lax">2.96</td>
    <td class="tg-0lax">0.03</td>
    <td class="tg-0lax">2.71</td>
    <td class="tg-0lax">1.66</td>
    <td class="tg-0lax">78.60</td>
    <td class="tg-0lax">6.48</td>
  </tr>
  <tr>
    <td class="tg-0lax">RWKV-4-14B-pile</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax">0.71</td>
    <td class="tg-0lax">9.52E-307</td>
    <td class="tg-0lax">18.54</td>
    <td class="tg-0lax">1.55</td>
    <td class="tg-0lax">17.69</td>
    <td class="tg-0lax">3.45</td>
    <td class="tg-0lax">71.36</td>
    <td class="tg-0lax">5.33</td>
  </tr>
  <tr>
    <td class="tg-0lax">LLaMA2-7B-32K</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">0.08</td>
    <td class="tg-0lax">2.44E-308</td>
    <td class="tg-0lax">2.05</td>
    <td class="tg-0lax">0.00</td>
    <td class="tg-0lax">2.05</td>
    <td class="tg-0lax">0.46</td>
    <td class="tg-0lax">50.28</td>
    <td class="tg-0lax">4.18</td>
  </tr>
</tbody>
</table>

<br>

### **Impact of input length on long dependency tasks**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">Models </th>
    <th class="tg-0lax">Context </th>
    <th class="tg-0lax">Bleu1</th>
    <th class="tg-0lax"> Bleu4 </th>
    <th class="tg-0lax">Rouge1 </th>
    <th class="tg-0lax">Rouge4 </th>
    <th class="tg-0lax">RougeL </th>
    <th class="tg-0lax">Meteor score </th>
    <th class="tg-0lax">Bert score </th>
    <th class="tg-0lax">GPT4 score </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax" colspan="10">arXiv paper summarization</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">24.50</td>
    <td class="tg-0lax">0.73</td>
    <td class="tg-0lax">27.15</td>
    <td class="tg-0lax">7.10</td>
    <td class="tg-0lax">24.25</td>
    <td class="tg-0lax">19.03</td>
    <td class="tg-0lax">84.04</td>
    <td class="tg-0lax">82.84</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">24k</td>
    <td class="tg-0lax">25.57</td>
    <td class="tg-0lax">0.81</td>
    <td class="tg-0lax">27.61</td>
    <td class="tg-0lax">7.53</td>
    <td class="tg-0lax">24.73</td>
    <td class="tg-0lax">19.86</td>
    <td class="tg-0lax">84.07</td>
    <td class="tg-0lax">83.15</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">16k</td>
    <td class="tg-0lax">24.8</td>
    <td class="tg-0lax">0.70</td>
    <td class="tg-0lax">27.29</td>
    <td class="tg-0lax">7.26</td>
    <td class="tg-0lax">24.28</td>
    <td class="tg-0lax">19.12</td>
    <td class="tg-0lax">84.11</td>
    <td class="tg-0lax">82.82</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax">26.26</td>
    <td class="tg-0lax"><b>9.35</b></td>
    <td class="tg-0lax">27.83</td>
    <td class="tg-0lax">7.67</td>
    <td class="tg-0lax">24.74</td>
    <td class="tg-0lax">20.08</td>
    <td class="tg-0lax">84.10</td>
    <td class="tg-0lax">82.75</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-8k</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax"><b>29.02</b></td>
    <td class="tg-0lax">2.09</td>
    <td class="tg-0lax"><b>32.08</b></td>
    <td class="tg-0lax"><b>11.11</b></td>
    <td class="tg-0lax"><b>28.85</b></td>
    <td class="tg-0lax"><b>22.64</b></td>
    <td class="tg-0lax"><b>84.92</b></td>
    <td class="tg-0lax"><b>85.42</b></td>
  </tr>
  <tr>
    <td class="tg-0lax" colspan="10">Long dependency QA</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">32k</td>
    <td class="tg-0lax">7.64</td>
    <td class="tg-0lax">1.24</td>
    <td class="tg-0lax">15.53</td>
    <td class="tg-0lax">4.46</td>
    <td class="tg-0lax">14.60</td>
    <td class="tg-0lax">11.12</td>
    <td class="tg-0lax">86.07</td>
    <td class="tg-0lax"><b>54.65</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">24k</td>
    <td class="tg-0lax">8.23</td>
    <td class="tg-0lax">1.66</td>
    <td class="tg-0lax">14.92</td>
    <td class="tg-0lax">4.12</td>
    <td class="tg-0lax">13.90</td>
    <td class="tg-0lax">10.60</td>
    <td class="tg-0lax">86.16</td>
    <td class="tg-0lax">50.61</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">16k</td>
    <td class="tg-0lax">8.57</td>
    <td class="tg-0lax">1.35</td>
    <td class="tg-0lax">16.21</td>
    <td class="tg-0lax">4.30</td>
    <td class="tg-0lax">14.90</td>
    <td class="tg-0lax"><b>11.91</b></td>
    <td class="tg-0lax"><b>86.36</b></td>
    <td class="tg-0lax">47.55</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-32k</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax">7.46</td>
    <td class="tg-0lax"><b>1.77</b></td>
    <td class="tg-0lax">13.75</td>
    <td class="tg-0lax">5.08</td>
    <td class="tg-0lax">12.89</td>
    <td class="tg-0lax">10.01</td>
    <td class="tg-0lax">85.77</td>
    <td class="tg-0lax">38.34</td>
  </tr>
  <tr>
    <td class="tg-0lax">GPT4-8k</td>
    <td class="tg-0lax">8k</td>
    <td class="tg-0lax"><b>8.94</b></td>
    <td class="tg-0lax">1.01</td>
    <td class="tg-0lax"><b>23.45</b></td>
    <td class="tg-0lax"><b>6.57</b></td>
    <td class="tg-0lax"><b>21.69</b></td>
    <td class="tg-0lax">10.18</td>
    <td class="tg-0lax">85.36</td>
    <td class="tg-0lax">42.12</td>
  </tr>
</tbody>
</table>

<!-- ## 📝 **Tools**
Here is an example for our annotation websit for long dependency QA task.
<br> -->

## 📝 **Citation**
If you would like to use our data or find our work interesting, please cite:
```bibtex
@misc{li2023loogle,
  title={Can Long-Context Language Models Understand Long Contexts?},
  author={ Li, Jiaqi and Wang, Mengmeng and Zheng, Zilong and Zhang, Muhan },
  url={https://github.com/bigai-nlco/LooGLE},
  year={2023}
}
```

## 📣 **Contacts**

We sincerely appreciate human annotators for their valuable contributions on creating high-quality long-dependency QA tasks.
We are very pleased to answer any questions about LooGLE: [nlp@bigai.ai](mailto:nlp@bigai.ai)
