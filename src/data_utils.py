import json
import pandas as pd
from tqdm import tqdm
import sys

def read_data(file_name):
    items = []
    for i in open(file_name,'r').readlines():
        items.append(json.loads(i))
    return pd.DataFrame(items)
def save_data(df, o_name):
    df = df.astype(object)
    with open(f"{o_name}.json",'w+') as t:
        for i in tqdm(range(len(df))):
            item = df.iloc[i,:].to_dict()
            t.write(json.dumps(item)+'\n')
def save_dict(d, o_name):
    with open(f"{o_name}.json",'w+') as o:
        o.write(json.dumps(d))

def get_example(data, index):
    return data.iloc[index,:].to_dict()

import os
def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
def get_rank():
    return int(os.environ.get("RANK", "0"))
def get_world_size():
    return os.environ.get("CUDA_VISIBLE_DEVICES","0").count(',')+1

import copy

# Get Data
def get_data(data_dir):
    base_dir=data_dir #'/raid/lqy/Mutation/Classifier/Dataset/RealEdits'
    train = read_data(f"{base_dir}/train.json")#.iloc[:500,:]
    valid = read_data(f"{base_dir}/valid.json")#.iloc[:100,:]
    test = read_data(f"{base_dir}/test.json")#.iloc[:100,:]
    return train, valid, test

def get_pos_data(data_dir):
    base_dir=data_dir
    train = read_data(f"{base_dir}/train.json")
    train = train.loc[train.label==1]
    valid = read_data(f"{base_dir}/valid.json")
    valid = valid.loc[valid.label==1]
    test = read_data(f"{base_dir}/test.json")
    test = test.loc[test.label==1]
    return train, valid, test

import re
def tokenize(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens
def remove_python_comments(code):
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    code = re.sub(r'(#.*)|((?<=\s)#.*)', '', code)
    code = "\n".join([c for c in code.splitlines() if len(c.split())!=0])
    return code.strip()

def code_preprocess(code):
    return ' '.join(tokenize(code))
def nl_preprocess(nl):
    return '\n'.join([i for i in nl.split('\n') if len(i)!=0])

# Dataset
from torch.utils.data import Dataset
import torch
class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, max_len=512, data_type='train'):
        self.tokenizer = tokenizer
        self.data = read_data(f"{args.data_dir}/{data_type}.json")#.iloc[:500,:]
        print("withNL: Training with natural language......")
        # Please determine whether this code is the correct solution for the problem description.
        source_pos = [j+"\n\n\n"+i for i,j in zip(self.data.nl.tolist(), self.data.pos.tolist())]
        source_neg = [j+"\n\n\n"+i for i,j in zip(self.data.nl.tolist(), self.data.neg.tolist())]
        self.sources = source_pos + source_neg
        self.targets = [1]*len(source_pos) + [0]*len(source_neg)
        
        self.max_len = max_len

        print("example of src text: ", self.sources[10])
        print("example of tgt label: ", self.targets[10])

        if data_type=='train':
            world_size = get_world_size()
            local_rank = get_rank()

            savep = f"{args.cache_path}/{args.task.split('_')[0]}_{data_type}_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank) + ".exps"
            if os.path.exists(savep):
                print("Loading examples from {}".format(savep))
                try:
                    self.feats = torch.load(savep)
                except Exception as e:
                    print(f"Loading Failed")
                    raise(e)
            else:
                self.feats = []
                if len(self.sources)%world_size!=0:
                    self.sources = self.sources[:-(len(self.sources)%world_size)]
                    self.targets = self.targets[:-(len(self.targets)%world_size)]
                # all features
                all_feats = []
                for idx in tqdm(range(len(self.sources)), total=len(self.sources)):
                    fs = self.tokenize(idx)
                    all_feats.append(fs)
                print(f"******* length of all feats: ******* : ", len(all_feats))
                for idx in tqdm(range(len(all_feats)), total=len(all_feats)):
                    if idx % world_size != local_rank:
                        continue
                    self.feats.append(all_feats[idx])
                print(f"******* length of feats in rank {local_rank} ******* : ", len(self.feats))
                torch.save(self.feats, savep)
        else:
            self.feats = [self.tokenize(idx) for idx in range(len(self.sources))]

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

    def tokenize(self, index):
        source = str(self.sources[index])
        inputs = self.tokenizer.encode_plus(
            source,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        if index==0:
            print("example of ids: ", ids)
            print("example str of ids: ", self.tokenizer.decode(ids))

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target
                 ):
        self.idx = idx
        self.source = source
        self.target = target

class CLDataset(Dataset):
    def __init__(self, args, tokenizer, max_len=512, data_type='train_cl_512_sample500'):
        self.tokenizer = tokenizer
        print(f"data path: {args.data_dir}/{data_type}")
        self.data = read_data(f"{args.data_dir}/{data_type}.json")#.iloc[:500,:]

        self.nl_list = self.data.nl.tolist()
        self.code1_list = self.data.code1.tolist()
        self.code2_list = self.data.code2.tolist()
        self.label = self.data.label.tolist()
        self.max_len = max_len

        print("length of self.pos_sources: " ,len(self.code1_list))
        assert len(self.nl_list) == len(self.code1_list)

        world_size = get_world_size()
        local_rank = get_rank()

        cache_dir = args.cache_path
        os.makedirs(cache_dir, exist_ok=True)
        savep = f"{cache_dir}/{args.task.split('_')[0]}_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank) + ".exps"
        if os.path.exists(savep):
            print("Loading examples from {}".format(savep))
            try:
                self.feats = torch.load(savep)
            except Exception as e:
                print(f"Loading Failed")
                raise(e)
        else:
            print("Loading examples from {}".format(args.data_dir))
            self.feats = []
            data_length = len(self.nl_list)
            if data_length%world_size!=0:
                self.nl_list = self.nl_list[:-(data_length%world_size)]
                self.code1_list = self.code1_list[:-(data_length%world_size)]
                self.code2_list = self.code2_list[:-(data_length%world_size)]
                self.label = self.label[:-(data_length%world_size)]
            # all features
            all_feats = []
            for idx in tqdm(range(len(self.nl_list)), total=len(self.nl_list)):
                fs = self.tokenize(idx)
                if len(fs['ids'])<=self.max_len:
                    all_feats.append(fs)
            print(f"******* length of all feats: ******* : ", len(all_feats))
            for idx in tqdm(range(len(all_feats)), total=len(all_feats)):
                if idx % world_size != local_rank:
                    continue
                self.feats.append(all_feats[idx])
            print(f"******* length of feats in rank {local_rank} ******* : ", len(self.feats))
            torch.save(self.feats, savep)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]
    
    def tokenize(self, index):
        nl = str(self.nl_list[index])
        code1 = str(self.code1_list[index])
        code2 = str(self.code2_list[index])

        inputs = self.tokenizer.encode_plus(
            nl,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']

        inputs_code1 = self.tokenizer.encode_plus(
            code1,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids_code1 = inputs_code1['input_ids']

        inputs_code2 = self.tokenizer.encode_plus(
            code2,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids_code2 = inputs_code2['input_ids']

        # inputs_code1 = self.tokenizer.encode_plus(
        #     nl+ '\n\n' +code1,
        #     None,
        #     truncation=True,
        #     padding='max_length',
        #     add_special_tokens=True,
        #     max_length=self.max_len+self.max_len//2,
        #     return_token_type_ids=True
        # )
        # ids_code1 = inputs_code1['input_ids']

        # inputs_code2 = self.tokenizer.encode_plus(
        #     nl+ '\n\n' +code2,
        #     None,
        #     truncation=True,
        #     padding='max_length',
        #     add_special_tokens=True,
        #     max_length=self.max_len+self.max_len//2,
        #     return_token_type_ids=True
        # )
        # ids_code2 = inputs_code2['input_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'targets': torch.tensor(self.label[index], dtype=torch.float),
            'ids_code1': torch.tensor(ids_code1, dtype=torch.long),
            'ids_code2': torch.tensor(ids_code2, dtype=torch.long),
        }


class DatasetCA(Dataset):
    def __init__(self, args, tokenizer, max_len=512, data_type='train'):
        self.tokenizer = tokenizer
        print(f"data path: {args.data_dir}/{data_type}")
        self.data = read_data(f"{args.data_dir}/{data_type}.json")#.iloc[:500,:]
        
        if 'pos' in self.data.columns:
            self.nl = self.data.nl.tolist() + self.data.nl.tolist()
            self.code = self.data.pos.tolist() + self.data.neg.tolist()
            self.targets = [1]*len(self.data.nl.tolist()) + [0]*len(self.data.nl.tolist())
            assert len(self.nl) == len(self.code) == len(self.targets)
        else:
            self.nl = self.data.nl.tolist()
            self.code = self.data.code.tolist()
            self.targets = self.data.label.tolist()

        self.max_len = max_len

        print("length of self.code: " ,len(self.code))
        print("example of nl text: ", self.nl[10])
        print("example of code text: ", self.code[10])
        print("example of target label: ", self.targets[10])

        if data_type=='train':
            world_size = get_world_size()
            local_rank = get_rank()
            cache_dir = args.cache_path
            os.makedirs(cache_dir, exist_ok=True)
            savep = f"{cache_dir}{args.task.split('_')[0]}_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank) + ".exps"
            if os.path.exists(savep):
                print("Loading examples from {}".format(savep))
                try:
                    self.feats = torch.load(savep)
                except Exception as e:
                    print(f"Loading Failed")
                    raise(e)
            else:
                print("Loading examples from {}".format(args.data_dir))
                self.feats = []
                data_length = len(self.code)
                if data_length%world_size!=0:
                    self.nl = self.nl[:-(data_length%world_size)]
                    self.code = self.code[:-(data_length%world_size)]
                    self.targets = self.targets[:-(data_length%world_size)]
                all_feats = []
                for idx in tqdm(range(len(self.code)), total=len(self.code)):
                    fs = self.tokenize(idx)
                    all_feats.append(fs)
                print(f"******* length of all feats: ******* : ", len(all_feats))
                for idx in tqdm(range(len(all_feats)), total=len(all_feats)):
                    if idx % world_size != local_rank:
                        continue
                    self.feats.append(all_feats[idx])
                print(f"******* length of feats in rank {local_rank} ******* : ", len(self.feats))
                torch.save(self.feats, savep)
        else:
            self.feats = [self.tokenize(idx) for idx in range(len(self.code))]
    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]
    
    def tokenize(self, index):
        code = str(self.code[index])
        nl = str(self.nl[index])

        inputs = self.tokenizer.encode_plus(
            nl,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len//2,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        inputs_2 = self.tokenizer.encode_plus(
            code,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids_2 = inputs_2['input_ids']
        mask_2 = inputs_2['attention_mask']

        if index==0:
            print("example of ids: ", ids)
            print("example str of ids: ", self.tokenizer.decode(ids))
            print("example of ids: ", ids_2)
            print("example str of ids: ", self.tokenizer.decode(ids_2))

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            # 'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'ids_2': torch.tensor(ids_2, dtype=torch.long),
            # 'mask_2': torch.tensor(mask_2, dtype=torch.long),
        }
