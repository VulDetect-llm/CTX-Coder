
<a name="readme-top"></a>



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">CTX-Coder: Cross-Attention Architectures Empower LLMs for Long-Context Vulnerability Detection</h3>

  <p align="center">
    A Long-Context Enhancing LLM For Vulnerability Detection. 
    <br />
    <a href=""><strong>Explore the docs »</strong></a>
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
> The dataset and checkpoint is comming soon!


## CTX-Coder
### Training
We provide the training scripts in ctx_coder/train_ctxcoder.py, to use this script please fill the `MODEL_PATH`, `LLAMA_3_PATH`, and `OUTPUT_PATH`.
You can train the model using the following command:
```
deepspeed ctx_coder/train_ctxcoder.py
``` 

### Inference
We provide a pipeline, you can just replace the trained checkpoint and dataset for inference. Using the following command:
```
python ctx_coder/pipeline.py
```

## Evaluation
- To evaluate CTX-Coder, you should generate fisrst the result using `pipeline.py`. Then evaluate the result using `evaluation/test.py`.

- For code document generation, we use the default dataset of [CodeBert](https://github.com/microsoft/CodeBERT/tree/master) and use the official code of [Big-Code](https://github.com/bigcode-project/bigcode-evaluation-harness).

- CrossCodeEval: project url https://github.com/amazon-science/cceval.





