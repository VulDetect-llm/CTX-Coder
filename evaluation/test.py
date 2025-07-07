import json
import random
from sklearn.metrics import accuracy_score, f1_score
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from tqdm import tqdm

def truncate(prompt: str, max_num_tokens: int, side: str, tokenizer) -> str:
    """Truncate prompt from side given the token budget"""

    tokens = tokenizer.tokenize(prompt)
    num_tokens = len(tokens)

    if num_tokens > max_num_tokens:
        if side == 'left':
            prompt_tokens = tokens[num_tokens - max_num_tokens:]
        elif side == 'right':
            prompt_tokens = tokens[:max_num_tokens]
        prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
    return prompt

def extract_answer(text, d):
    try:
        response_cwe = re.search(r'This function .* has (CWE-.*?):', text, re.DOTALL)
        if response_cwe:
            if d['vul_type'] == 'Vulnerable':
                cwe_id = response_cwe.group(1)
                if cwe_id == d['cwe_id']:
                    return 1
                else:
                    return -1
            else:
                return 1
        else:
            return 0
    except Exception:
        print("error")
        return -1

def test_llm(model_path, data_path):
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side  = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path, 'r') as f:
        data = json.load(f)

    
    detect_prompt = open('prompts/detect_vul.txt', 'r').read()

    prompts = []
    ground_truths = []

    for d in tqdm(data):
        if 'vul_code' in d:
            code = d['vul_code']
        else:
            code = d['index_to_code']["0"]
    
        index_to_code = d['index_to_code']
        remaining_codes = [
            index_to_code[str(i)]
            for i in range(1, len(index_to_code))
            if str(i) in index_to_code and index_to_code[str(i)] is not None
        ]

        if remaining_codes:
            # 截断每段代码，确保是字符串
            remaining_codes = [
                truncate(code, 1024, 'right', tokenizer)
                for code in remaining_codes if isinstance(code, str)
            ]
            concatenated_code = "\n\n\n".join(remaining_codes)
        else:
            concatenated_code = "None"
        user_prompt = detect_prompt.replace('<code>', code).replace('<ctx>', concatenated_code)

        # 如果是聊天模型，构建完整对话上下文
        content = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': user_prompt},
             {'role': 'assistant', 'content': ''}],
            tokenize=False
        ).rsplit(tokenizer.eos_token, 1)[0]  # 去掉未完的 assistant 结束符，适合生成

        prompts.append(content)
        ground_truths.append(1 if d['vul_type'] == 'Vulnerable' else 0)

    # 初始化 vLLM 模型
    print("Loading vLLM model...")
    llm = LLM(model=model_path, tensor_parallel_size=2, dtype="bfloat16", gpu_memory_utilization=0.8)

    sampling_params = SamplingParams( 
        temperature=0.0,
        max_tokens=150,
        top_p=1.0,
        stop=[tokenizer.eos_token]
    )

    print("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)

    predictions = []
    for i, (output, d) in enumerate(zip(outputs, data)):
        output_text = output.outputs[0].text.strip()
        d['pred'] = output_text
        # print(f"\n--- Sample {i} Output ---\n{output_text}\n")
        pred = extract_answer(output_text, d)
        if pred == -1:
            pred = 1 - ground_truths[i]
        predictions.append(pred)

    # 计算指标
    acc = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # 打印 TP / FP / TN / FN
    tp = sum((p == 1 and g == 1) for p, g in zip(predictions, ground_truths))
    fp = sum((p == 1 and g == 0) for p, g in zip(predictions, ground_truths))
    tn = sum((p == 0 and g == 0) for p, g in zip(predictions, ground_truths))
    fn = sum((p == 0 and g == 1) for p, g in zip(predictions, ground_truths))

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    return acc, f1


def test_ctx():
    with open('prediction.jsonl', 'r') as f:
        data = [json.loads(i) for i in f.readlines()]
    predictions = []
    ground_truths = [1 if d['vul_type'] == 'Vulnerable' else 0 for d in data]
    for i, d in enumerate(data):
        output_text = d["pred"]
        # print(f"\n--- Sample {i} Output ---\n{output_text}\n")
        pred = extract_answer(output_text, d)
        if pred == -1:
            pred = 1 - ground_truths[i]
        predictions.append(pred)

    # 计算指标
    acc = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # 打印 TP / FP / TN / FN
    tp = sum((p == 1 and g == 1) for p, g in zip(predictions, ground_truths))
    fp = sum((p == 1 and g == 0) for p, g in zip(predictions, ground_truths))
    tn = sum((p == 0 and g == 0) for p, g in zip(predictions, ground_truths))
    fn = sum((p == 0 and g == 1) for p, g in zip(predictions, ground_truths))

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    return acc, f1


if __name__ == '__main__':
    test_ctx()