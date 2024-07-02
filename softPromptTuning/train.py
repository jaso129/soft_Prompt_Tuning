import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import dataset
from itertools import cycle
from copy import deepcopy
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from sklearn.metrics import matthews_corrcoef, f1_score



class T5ContinualLearner:
    def __init__(self,
                 model_name,
                 task_list,
                 batch_size=8,
                 select_k_per_class=-1,
                 prefix_len=0,
                 prefix_path=None, # path to the pre-trained progressive prompt
                 freeze_weights=True,
                 freeze_except='shared',
                 lr=0.3,
                #  weight_decay=1e-5,
                 seq_len=512,
                 early_stopping=True,
                #  prefix_MLP='None',
                #  bottleneck_size=800, # bottleneck size in case of using MLP reparametrization
                #  mlp_lr=None,
                #  mlp_layer_norm=False,
                #  weight_decay_mlp=None,
                 get_test_subset=True,
                 memory_perc=0.0,
                 ):
        
        """Class for CL & prompt tuning experiments with T5 model.
        Args:
            model_name (str): T5 model type to use (e.g. base/small/large etc.)
            task_list (List[str]): list of downstream tasks to be trained on. In case of 1 task - regular training.
            batch_size (int, optional): Batch size used. Defaults to 8.
            select_k_per_class (int, optional): Limit data to k samples/class. Defaults to -1 (keep original dataset size).
            prefix_len (int, optional): Prompt length to use. Defaults to 0 (i.e. no prompt).
            prefix_path (str, optional): Path to the pre-trained progressive prompt. Defaults to None.
            freeze_weights (bool, optional): Whether to freeze model weights. Defaults to True (prompt tuning setup).
            freeze_except (str, optional): Freeze all weights except parameters matching this condition. 
                Defaults to 'shared' (freeze all weights except word embeddings).
            lr (float, optional): Learning rate. Defaults to 0.3.
            weight_decay (float, optional): Weight decay coefficient. Defaults to 1e-5.
            seq_len (int, optional): Input text lengths in tokens. Defaults to 512.
            early_stopping (bool, optional): Use early stopping to select best prompt/model. Defaults to True.
            prefix_MLP (str, optional): what MLP to use for prompt re-parameterization. Defaults to 'MLP-1'.
            bottleneck_size (int, optional): Bottleneck size in case of using MLP reparametrization. Defaults to 800.
            mlp_lr (float, optional): MLP learning rate to use. Defaults to None (lr value will be used).
            weight_decay_mlp (float, optional): Wight decay coefficient in MLP. Defaults to None.
            get_test_subset (bool, optional): Whether to create a test subset. Defaults to True.
            memory_perc (float, optional): Percentage of data saved for memory replay in CL settings. Defaults to 0.0.
                 
                 
            prefix_len (int, optional): Soft prompt length (only needed if virtual tokens are added to the vocab). Defaults to 0.
            freeze_weights (bool, optional): Whether to freeze base model weights. 
                Model weights need to be frozen for prompt tuning (i.e. True)! Defaults to False.
            freeze_except (str, optional): If freeze_weights, do not freeze weights that contain this text. 
                Defaults to 'shared' (will avoid freezing word embeddings layer in T5).
            lr (float, optional): Prompt (model) learning rate. Defaults to 0.1.
            weight_decay (float, optional): Prompt (model) weight decay coefficient. Defaults to 0.00.
            prompt_name (str, optional): Shared name for prompt virtual tokens (when added to the vocab). 
                Not used in the latest implementation. Defaults to 'PRE'.
            
            prefix_MLP (str, optional): . Defaults to 'None'.
            mlp_bottleneck (int, optional): MLP bottleneck dimension. Defaults to 1000.
            weight_decay_mlp (float, optional): MLP weight decay coefficient. Defaults to 0.01.
            mlp_lr (float, optional): MLP learning rate. Defaults to 1e-4.
            mlp_layer_norm (bool, optional): Whether to use LN in MLP. Defaults to True.
            
            early_stopping (bool, optional): Whether to select best paramteres via early stopping. Defaults to True.
            opt (str, optional): Optimizer to use. Curretnly AdamW and LAMB are supported. Defaults to 'AdamW'.
        
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used. 
                Currently supports 1-layer and 2-layer MLPs ('MLP1' and 'MLP2'). Defaults to 'MLP1'.
            emb_dimension (int, optional): . Defaults to 512.
            layer_norm (bool, optional): . Defaults to True.
        """
        
        
        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'wsc_bool', 'cb', 'record', 'multirc', 'rte_superglue']
        self.task_to_target_len = {
            'rte': 5,
            'mrpc': 5,
            'sst2': 2,
            'qqp': 5,
            'cola': 5,
            'qnli': 5,
            'mnli': 5,
            'stsb': 3,

            'wic': 2,
            'boolq': 2,
            'copa': 2,
            'wsc': 3,
            'wsc_bool': 2,
            'cb': 5,
            'multirc': 5,
            'record': 10,
            'rte_superglue': 5,

            'imdb': 2,

            'ag_news': 2,
            'yahoo_answers_topics': 5,
            'dbpedia_14': 5,
            'amazon': 2,
            'yelp_review_full': 2,
        }
        self.task_list = task_list

        self.freeze_weights = freeze_weights
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.select_k_per_class = select_k_per_class
        self.early_stopping = early_stopping

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print('使用cpu or cuda:',self.device)

        self.model_name = "t5-large" # e.g. "t5-large"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Freezing model weights for prompt tuning
        if freeze_weights:
            print('Freezing weights')
            self.do_freeze_weights(except_condition=freeze_except)
           
        self.prefix_len = prefix_len
        # Creating a trainable soft prompt
        if prefix_len>0:
            self.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(prefix_len),
                                                          requires_grad=True))
            if prefix_path==None:
                self.previous_prompts = torch.zeros([0, self.model.prompt.shape[1]],
                                                    requires_grad=False).to(self.device)
            else: # initializing previous prompts from the path
                print('Using pre-trained progressive prompt - ' + prefix_path)
                self.previous_prompts = torch.tensor(np.load(prefix_path), requires_grad = False).to(self.device)
        
    
        # Get task -> data dictionary for CL training
        self.get_test_subset = get_test_subset
        self.tasks_data_dict = self.get_tasks_data_dict(memory_perc=memory_perc)

        print('經過前處理後的資料集如下：\n',self.tasks_data_dict)


    def get_datasets_dataloader(self):
        return self.tasks_data_dict

    # Initialize new task prompt from random vocab. tokens
    def init_new_prompt(self, prompt_len):
        model = self.model
        N = model.encoder.embed_tokens.weight.shape[0]
        prompt_weigths = []

        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N) # random token
                w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weigths.append(w)
        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths

    # Create Dictionary of task_name -> dataloader (for CL experiments)
    def get_tasks_data_dict(self, memory_perc=0):
            tasks_data_dict = {} # 創建任務資料集字典，裡面儲存該任務的訓練、驗證、測試的dataloader

            for task in self.task_list: # 迭代多個任務做前處理
                tasks_data_dict[task] = {} # 在任務資料集字典中為每個任務在創建一個字典
                print(task)
                data_params = {'task': task,
                            'batch_size': self.batch_size,
                            'max_length': self.seq_len, # 輸入資料最大序列長度
                            'target_len': self.task_to_target_len[task], # 決定產生的輸出長度
                            'prefix_list': [], # we are using vector prefix (instead of tokenization)
                            }
                ds2 = dataset.T5Dataset(self.tokenizer, task)
                if task not in ['mrpc', 'cola', 'copa', 'rte', 'rte_superglue', 'cb', 'wsc', 'wsc_bool']:
                    k = self.select_k_per_class
                    k_val = max(500, int(0.2*k)) if task!='sst2' else 400
                else:
                    k = self.select_k_per_class if (self.select_k_per_class<=500 and task not in ['cb', 'copa', 'wsc', 'wsc_bool']) else -1
                    k_val = -1
                if self.get_test_subset==False: k_val = -1 # use all val set
                dataloader_train = ds2.get_final_ds(**data_params, k=k, split='train') # 得到訓練集
                print('k = ', k, '  k-val = ',k_val)
                val_split = 'validation' if (task in self.glue_datasets) or (task in self.superglue_datasets) else 'test' # 若任務不屬於glue or superglue 則驗證集為測試集
                dataloaders = ds2.get_final_ds(**data_params, k=k_val,
                                            split=val_split, return_test=self.get_test_subset)

                tasks_data_dict[task]['train'] = dataloader_train # 將訓練集dataloader存入任務字典

                # if memory_perc>0:
                #     k_mem = max(1, int(len(dataloader_train) * self.batch_size * memory_perc) )
                #     dataloader_mem = ds2.get_final_ds(**data_params, k=k_mem, split='train')
                #     tasks_data_dict[task]['train_mem'] = dataloader_mem

                if self.get_test_subset:
                    dataloader_val, dataloader_test = dataloaders[0], dataloaders[1]
                    tasks_data_dict[task]['val'] = dataloader_val
                    tasks_data_dict[task]['test'] = dataloader_test
                else:
                    tasks_data_dict[task]['val'] = dataloaders

                if task == 'multirc' and k_val==-1:
                    self.multirc_idx = ds2.multirc_idx # saving multirc idx for later computation
                else: self.multirc_idx = None
            return tasks_data_dict
    

