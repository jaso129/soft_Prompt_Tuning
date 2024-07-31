import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from copy import deepcopy
import numpy as np


def init_new_prompt(prompt_len):
        model_name='t5-small'
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # 效能理論上會好過從預訓練模型的詞彙表中提取詞嵌入當作soft prompt
        # 嘗試離散提示集合轉為連續
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        prompt_list = [
                'Assess the entire concept of the news story and choose from the World, Sports, Business or Tech categories to categorize it into the correct category.',
        ] 
        soft_prompts=[]
        selected_prompts = np.random.choice(prompt_list, size=1, replace=False)
        token_ids = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in selected_prompts]
        max_length = max(len(ids) for ids in token_ids)
        padding_token_id = tokenizer.pad_token_id
        padded_token_ids = [ids + [padding_token_id] * (max_length - len(ids)) for ids in token_ids]
        embeddings = model.get_input_embeddings()
        soft_prompts = deepcopy([embeddings(torch.tensor(ids)).detach().cpu().numpy() for ids in padded_token_ids])
        soft_prompts=np.array(soft_prompts)
        print(soft_prompts.shape)
        soft_prompts = np.mean(soft_prompts, axis=1)



        N = model.encoder.embed_tokens.weight.shape[0] # get vocab size (32128)ˋ
        soft_prompts2 = []
        # print('embedding layer shape:',model.encoder.embed_tokens)

        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N) # random token
                w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy()) # copy random embedding become soft prompt
                soft_prompts2.append(w)

        soft_prompts2 = np.array(soft_prompts2)
        # print('prompt_weigths.shape:', prompt_weigths.shape)

        print('soft prompt shape:', soft_prompts.shape) 
        print('soft_prompts2 shape:', soft_prompts2.shape)

        return soft_prompts

softPrompt=nn.Parameter(torch.tensor(init_new_prompt(2),requires_grad=True))



