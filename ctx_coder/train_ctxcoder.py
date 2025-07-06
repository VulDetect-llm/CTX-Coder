from dataclasses import dataclass, field
import json
import os
from typing import Dict, Optional, List, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer
from transformers import AutoModel
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
import random
import warnings
from model import MllamaForConditionalGeneration, MllamaCrossAttentionDecoderLayer
import torch.nn.functional as F
import numpy as np
# from .trainer import MyCustomTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

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
        new_len = len(tokenizer.tokenize(prompt))
    return prompt

def get_cross_attention_token_mask(input_ids: List[int], image_token_id: int) -> List[List[int]]:
    """
    Generate a cross-attention token mask for image tokens in the input sequence.

    This function identifies the positions of image tokens in the input sequence and creates
    a mask that defines which subsequent tokens each image token should attend to.

    Args:
        input_ids (List[int]): A list of token ids representing the input sequence.
        image_token_id (int): The id of the token used to represent images in the sequence.

    Returns:
        List[List[int]]: A list of [start, end] pairs, where each pair represents the range
        of tokens an image token should attend to.

    Notes:
        - If no image tokens are present, an empty list is returned.
        - For a single image token, it attends to all subsequent tokens until the end of the sequence.
        - For multiple image tokens, each attends to tokens up to the next image token or the end of the sequence.
        - Consecutive image tokens are treated as a group and attend to all subsequent tokens together.
    """

    image_token_locations = [i for i, token in enumerate(input_ids) if token == image_token_id]

    if len(image_token_locations) == 0:
        return []

    # only one image present, unmask until end of sequence
    if len(image_token_locations) == 1:
        return [[image_token_locations[0], -1]]

    vision_masks = [[loc1, loc2] for loc1, loc2 in zip(image_token_locations[:-1], image_token_locations[1:])]

    # last image will attend to all subsequent text
    vision_masks.append([image_token_locations[-1], len(input_ids)])

    # if there are two or more consecutive vision tokens,
    # they should all attend to all subsequent
    # text present
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]

    return vision_masks


def convert_sparse_cross_attention_mask_to_dense(
    cross_attention_token_mask: List[List[List[int]]],
    num_tiles: List[List[int]],
    max_num_tiles: int,
    length: int,
) -> np.ndarray:
    """
    Convert the cross attention mask indices to a cross attention mask 4D array.

    This function takes a sparse representation of cross attention masks and converts it to a dense 4D numpy array.
    The sparse representation is a nested list structure that defines attention ranges for each image in each batch item.

    Args:
        cross_attention_token_mask (List[List[List[int]]]): A nested list structure where:
            - The outer list represents the batch dimension.
            - The middle list represents different images within each batch item.
            - The inner list contains pairs of integers [start, end] representing token ranges for each image.
        num_tiles (List[List[int]]): A nested list structure specifying the number of tiles for each image in each batch item.
        max_num_tiles (int): The maximum possible number of tiles.
        length (int): The total sequence length of the input.

    Returns:
        np.ndarray: A 4D numpy array of shape (batch_size, length, max_num_images, max_num_tiles)
            The array contains `1` where attention is allowed and `0` where it is not.

    Note:
        - Special handling is done for cases where the end token is -1, which is interpreted as attending to the end of the sequence.
    """

    batch_size = len(cross_attention_token_mask)
    max_num_images = max([len(masks) for masks in cross_attention_token_mask])

    cross_attention_mask = np.zeros(
        shape=(batch_size, length, max_num_images, max_num_tiles),
        dtype=np.int64,
    )

    for sample_idx, (sample_masks, sample_num_tiles) in enumerate(zip(cross_attention_token_mask, num_tiles)):
        for mask_idx, (locations, mask_num_tiles) in enumerate(zip(sample_masks, sample_num_tiles)):
            if len(locations) == 2:
                start, end = locations
                end = min(end, length)
                if end == -1:
                    end = length
                cross_attention_mask[sample_idx, start:end, mask_idx, :mask_num_tiles] = 1
    return cross_attention_mask

@dataclass
class CustomDataCollator:
    padding_value: int = 0  # 填充值，默认为 0，可根据你的 tokenizer 的 pad_token_id 设置
    max_func_num: int = 20
    max_len: int = 4096
    gte_model: transformers.PreTrainedModel = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 提取 input_ids、attention_mask 和 labels
        input_ids = [feature['input_ids'] for feature in features]
        attention_masks = [feature['attention_mask'] for feature in features]
        labels = [feature['labels'] for feature in features]
        code_ids = [feature['code_ids'] for feature in features]
        code_attention_masks = [feature['code_attention_mask'] for feature in features]
        # 找到批次中最大的序列长度
        max_length = min(max(len(ids) for ids in input_ids), self.max_len)

        # 对 input_ids 和 attention_mask 进行填充
        padded_input_ids = [
            torch.nn.functional.pad(ids, (0, max_length - len(ids)), value=self.padding_value)
            for ids in input_ids
        ]
        padded_attention_masks = [
            torch.nn.functional.pad(mask, (0, max_length - len(mask)), value=0)
            for mask in attention_masks
        ]
        padded_labels = [
            torch.nn.functional.pad(label, (0, max_length - len(label)), value=IGNORE_TOKEN_ID)
            for label in labels
        ]

        # 对code_ids和code_attention_mask进行编码
        # 对code进行编码
        code_ids_output = []
        for idx in range(len(code_ids)):
            item = []
            code_id = code_ids[idx].to(f'cuda:{local_rank}')
            code_attention_mask = code_attention_masks[idx].to(f'cuda:{local_rank}') 
            # 两两一组进行batch编码
            for b in range(0, code_id.shape[0], 2):
                b_code_id = code_id[b:b+2]
                b_code_attention_mask = code_attention_mask[b:b+2]    
                with torch.no_grad():
                    item.append(self.gte_model(b_code_id, attention_mask=b_code_attention_mask, output_hidden_states=True).hidden_states[-1])
            item = torch.cat(item, dim=0)
            code_ids_output.append(item)
            


        # 将列表转换为张量
        batch_input_ids = torch.stack(padded_input_ids)
        batch_attention_masks = torch.stack(padded_attention_masks)
        batch_labels = torch.stack(padded_labels)
        adj_list = [f['adj'] for f in features if 'adj' in f]
        
        # 增加cross attention mask

        cross_attention_mask =  [
            get_cross_attention_token_mask(token_ids, 128256) for token_ids in batch_input_ids
        ]
        num_tiles = []
        for i in range(len(code_ids_output)):
            n = code_ids_output[i].shape[0]
            num_tiles.append([n])
        cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
            cross_attention_token_mask=cross_attention_mask,
            num_tiles=num_tiles,
            max_num_tiles=self.max_func_num,
            length=max_length
        )
        cross_attention_mask = torch.tensor(cross_attention_mask)
        # 对code_ids_output进行填充到self.max_func_num个函数
        for i in range(len(code_ids_output)):
            code_id = code_ids_output[i]
            n = code_id.shape[0]
            if n < self.max_func_num:
                padding_size = self.max_func_num - n
                code_id = F.pad(code_id, (0, 0, 0, 0, 0, padding_size))
                
            elif n > self.max_func_num:
                code_id = code_id[:self.max_func_num, :, :]
            code_ids_output[i] = code_id


        code_ids_output = torch.stack(code_ids_output).to(f'cuda:{local_rank}')
        code_input = code_ids_output.to('cpu')  # [bsz, 10, 1024, 1024]
        
        del code_ids, code_attention_masks, adj_list, code_ids_output
        # 返回批次字典
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'cross_attention_mask': cross_attention_mask,
            'labels': batch_labels,
            'cross_attention_states': code_input
        }


local_rank = None

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if trainer.args.use_lora:
        state_dict = get_peft_state_maybe_zero_3(
            trainer.model.named_parameters(), bias
        )
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    max_func_num: int = 10


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

def preprocess(data, tokenizer=None):
    ret = {}
    with open('prompts/detect_vul.txt', 'r') as f:
        task_prompt = f.read()
        
    # #### pretrain ####
    # code = data['index_to_code']['0']
    # prompt = f"<|image|><|begin_of_text|>{code}<|eot_id|>"
    # ret['text'] = prompt
    # ret['index_to_function_code'] = data['index_to_code']
    # ret['adj'] = data['adj']
        
        
    #### contextual eval ####
    prompt = task_prompt
    response = data['reason']
    if "vul_code" in data.keys():
        code = data['vul_code']
    else:
        code = data['index_to_code']['0']
    prompt = prompt.replace('<code>', truncate(code, 2048, 'left', tokenizer))
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n<|image|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{response}<|eot_id|>"
    text = prompt
    ret['text'] = text
    if len((data['index_to_code'])) > 1:
        del data['index_to_code']["0"]
        new_index_to_code = {str(int(key)-1): value for key, value in data['index_to_code'].items()}
        ret['index_to_function_code'] = new_index_to_code
    else:
        ret['index_to_function_code'] = data['index_to_code']
    ret['adj'] = data['adj']        
    return ret

class LazyPretrainDataset(Dataset):
    def __init__(self, raw_data, llama_tokenizer, gte_tokenizer, ignore_index=-100, max_len=8192):
        super().__init__()
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.ignore_index = ignore_index  # 定义忽略索引
        self.llama_tokenizer = llama_tokenizer
        self.gte_tokenizer = gte_tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        
        ret = preprocess(self.raw_data[i], self.llama_tokenizer)
        text = ret['text']
        tokenized = self.llama_tokenizer(text, max_length=8192, truncation=True)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        labels = torch.tensor(input_ids.copy(), dtype=torch.long)
        labels[labels == self.llama_tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        labels[labels == 128256] = IGNORE_TOKEN_ID
        index_to_function_code = [ret['index_to_function_code'][key] for key in sorted(ret['index_to_function_code'])]
        code_tokenized = self.gte_tokenizer(index_to_function_code, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': labels,
            'adj': torch.tensor(ret['adj']),
            'code_ids': code_tokenized.input_ids,
            'code_attention_mask': code_tokenized.attention_mask
        }
        self.cached_data_dict[i] = item
        return item

class LazySupervisedDataset(Dataset):
    def __init__(self, raw_data, llama_tokenizer, gte_tokenizer, ignore_index=-100, max_len=8192):
        super().__init__()
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.ignore_index = ignore_index  # 定义忽略索引
        self.llama_tokenizer = llama_tokenizer
        self.gte_tokenizer = gte_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        
        ret = preprocess(self.raw_data[i], self.llama_tokenizer)
        text = ret['text']
        
        # 定义助手回复的模板
        response_template = '<|start_header_id|>assistant<|end_header_id|>'
        
        # 对整个文本进行分词
        tokenized = self.llama_tokenizer(text, add_special_tokens=False, max_length=5000, truncation=True)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        labels = input_ids.copy()
        
        # 获取助手回复模板的 token IDs
        response_token_ids = self.llama_tokenizer(response_template, add_special_tokens=False).input_ids
        
        # 初始化助手回复的起始索引
        response_start_idx = None
        
        # 在输入 IDs 中查找助手回复模板的位置
        for idx in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[idx: idx + len(response_token_ids)] == response_token_ids:
                response_start_idx = idx + len(response_token_ids)
                break
        
        if response_start_idx is None:
            # 如果未找到助手回复模板，则将所有标签设置为 ignore_index，并发出警告
            labels = [IGNORE_TOKEN_ID] * len(labels)
            warnings.warn(
                f"Could not find response key `{response_template}` in the "
                f"following instance: {self.llama_tokenizer.decode(input_ids)} "
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider checking your data."
            )
        else:
            # 在助手回复之前的部分，标签设置为 ignore_index
            labels[:response_start_idx] = [IGNORE_TOKEN_ID] * response_start_idx
        index_to_function_code = [ret['index_to_function_code'][key] for key in sorted(ret['index_to_function_code'])]
        code_tokenized = self.gte_tokenizer(index_to_function_code, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        # 准备返回的数据
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'adj': torch.tensor(ret['adj']),
            'code_ids': code_tokenized.input_ids,
            'code_attention_mask': code_tokenized.attention_mask
        }
        
        # 缓存已处理的数据
        self.cached_data_dict[i] = item
        return item


def train():
    global local_rank
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    training_args = TrainingArguments(
        bf16=True,
        deepspeed='ds_config.json',
        output_dir=OUTPUT_DIR, 
        per_device_train_batch_size=2,
        num_train_epochs=2,
        seed=42,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        lr_scheduler_type='cosine',
        weight_decay=0.1,
        save_total_limit=3,
        save_strategy='steps',
        save_steps=5000,
        logging_steps=1,
        model_max_length=1024,
        cache_dir=CACHE_DIR,
        remove_unused_columns=False,
        label_names=['input_ids', 'attention_mask', 'labels', 'cross_attention_mask', 'cross_attention_state'],
    )
    local_rank = training_args.local_rank

    # 每次训练要修改
    model = MllamaForConditionalGeneration.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)
    gte_model = AutoModel.from_pretrained(LLAMA_3_PATH, torch_dtype=torch.bfloat16, device_map=f'cuda:{local_rank}')
    gte_model.eval()
    model.half()
    llama_tokenizer = transformers.AutoTokenizer.from_pretrained(
        LLAMA_3_PATH,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    gte_tokenizer = llama_tokenizer

    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

    for param in model.base_model.parameters():
        param.requires_grad = False
    
    
    for name, module in model.named_modules():
        if isinstance(module, MllamaCrossAttentionDecoderLayer):  # 判断是否是 MllamaCrossAttentionDecoderLayer 类型
            for param in module.parameters():
                param.requires_grad = True
    
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
    
    
    data = json.load(open('cve_data/ctx-eval-train.json'))
    train_data = data

    ## !!!每次训练前要改这个 LazyPretrainDataset 或 LazySupervisedDataset
    train_dataset = LazyPretrainDataset( 
        train_data,
        llama_tokenizer,
        gte_tokenizer,
        ignore_index=IGNORE_TOKEN_ID,
        max_len=training_args.model_max_length,
    )
    
    collator = CustomDataCollator(llama_tokenizer.pad_token_id, training_args.max_func_num, 1024, gte_model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=llama_tokenizer,
        data_collator=collator
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)


if __name__ == "__main__":
    MODEL_PATH = ""
    OUTPUT_DIR = ""
    LLAMA_3_PATH = "meta-llama-3.1-8b-instruct"
    CACHE_DIR = "cache"
    train()