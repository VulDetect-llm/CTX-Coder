<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/VulDetect-llm/CTX-Coder">
  </a>

<h3 align="center">CTX-Coder: Cross-Attention Architectures Empower LLMs for Long-Context Vulnerability Detection</h3>

  <p align="center">
    A Long-Context Enhancing LLM For Vulnerability Detection. 
    <br />
    <a href="https://github.com/VulDetect-llm/CTX-Coder"><strong>Explore the docs »</strong></a>
    <br />
    </p>
</div>



> [!NOTE]
> 
> The CTX-Coder is a modified version of [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
> We remove the vision encoder and use Llama's last hidden layers.




<!-- Quick Start -->
## Quick Install & Setup
```
# Install
pip3 install -r requirements.txt
```

## Call Graph Data Collection

If you want to collect your own call graph dataset, use the following steps:
1. Download the github projects into a directory `root`.
2. Generate the call graph using the following command:
```
cd doxygen
bash doxygen.sh

Note: please replace the root directory in doxygen.sh
```
3. Extract the function and format into json str using the python script: `python extract_doxygen.py`. It will output json files.


## CTX-VUL
The CTX-Vul dataset is a dataset contains contextual functions of a vulnerable function.
We format it in the following json string:
```json
{
    "index_to_funcname": {"0": "<func1_name>", "1": "<func2_name>"},
    "adj": ["# a n*n Matrix of the call relationships, A_{ij} = 1 means the function i is called by j"], 
    "index_to_code": {"0": "<func1_code>", "1": "<func2_code>"},
    "vul_type": "Vulnerable/Not Vulnerable"
}
```
> [!NOTE] 
> The  0 function is the target function.

The dataset can download from [CTX-Vul]()

## CTX-Coder
### Training
We provide the training scripts in ctx_coder/train_ctxcoder.py, to use this script please fill the `MODEL_PATH`, `LLAMA_3_PATH`, and `OUTPUT_PATH`.
You can train the model using the following command:
```
deepspeed ctx_coder/train_ctxcoder.py
``` 

### Inference
To inference the model, we provide a pipeline.



