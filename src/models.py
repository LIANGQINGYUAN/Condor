import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer, RobertaConfig, RobertaModel,
                          RobertaTokenizer, T5Config, T5EncoderModel, 
                          T5ForConditionalGeneration, T5Model, T5Tokenizer)
import torch.nn.functional as F
from torch import Tensor
from typing import Union
from transformers import AutoModel, AutoTokenizer, AutoConfig
logger = logging.getLogger(__name__)

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))



class CodeBERT(torch.nn.Module):
    def __init__(self, args, model_name, dim_text=768):
        super(CodeBERT, self).__init__()
        print("You are training with CodeBERT model.")
        self.args = args
        self.model = RobertaModel.from_pretrained(model_name, trust_remote_code=True)
        self.cross_att = CrossAttention(dim_text)
        self.fc = nn.Linear(dim_text*2, 1)
        self.loss = torch.nn.BCEWithLogitsLoss() 
        self.dense = nn.Linear(dim_text, 2)
        self.c_loss = ContrastiveLoss()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.dense.weight)

    def forward(
            self,
            text1=None,
            mask1=None,
            text2=None,
            mask2=None,
            text3=None,
            mask3=None,
            label1 = None,
            label2 = None,
            training_classifier=True,
    ):
        # training a classifier
        if training_classifier:
            enc_text1 = self.model(text1, attention_mask=mask1).last_hidden_state #nl [b, dim]
            enc_text2 = self.model(text2, attention_mask=mask2).last_hidden_state #code2 [b, dim]

            x1 = enc_text1.mean(dim=1) #self.to_text_latent(enc_text1.mean(dim=1))
            x2 = enc_text2.mean(dim=1) # self.to_text_latent2(enc_text2.mean(dim=1))

            # outputs
            outputs = self.fc(torch.cat([x1, x2],dim=-1))
            # loss
            if label1!=None:
                loss = self.loss(outputs, label1.view(-1,1).to(torch.float16))
            else:
                loss = None
            preds = torch.sigmoid(outputs.view(-1,1)) >= 0.5
            preds = torch.squeeze(preds).float()

            return preds, loss
        else:
            pass
            

class CodeT5ClassBCE(torch.nn.Module):
    def __init__(self, args, model_name):
        super(CodeT5ClassBCE, self).__init__()
        self.dim_text = 768
        # codet5 classification
        self.codeT5 = T5EncoderModel.from_pretrained(model_name)
        self.to_text_latent1 = nn.Linear(self.dim_text, self.dim_text)
        self.to_text_latent2 = nn.Linear(self.dim_text, self.dim_text)
        self.fc = nn.Linear(self.dim_text, 1)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.norm = nn.LayerNorm(self.dim_text)
        self.bnorm = nn.BatchNorm1d(self.dim_text)
    
    def forward(
            self,
            text1,
            mask1,
            label1 = None,
            training_classifier=True,
    ):
        # training a classifier
        if training_classifier:
            enc_text = self.codeT5(text1, attention_mask=mask1).last_hidden_state
            mask1_sum = mask1.sum(dim=-1, keepdims=True)
            enc_text = enc_text.masked_fill(mask1.unsqueeze(-1)==0, 0).sum(dim=1)/mask1_sum
            x1 = self.norm(self.to_text_latent1(enc_text))
            x1 = self.bnorm(self.to_text_latent2(x1))
            # outputs
            outputs = self.fc(x1)
            # loss
            if label1!=None:
                loss = self.loss(outputs.view(-1,1), label1.view(-1,1).to(torch.float16))
            else:
                loss = None
            preds = torch.sigmoid(outputs.view(-1,1)) >= 0.5
            preds = torch.squeeze(preds).float()
            return preds, loss



class DsClassBCE(torch.nn.Module):
    def __init__(self, args, model_name, dim_text=768):
        super(DsClassBCE, self).__init__()
        self.dim_text = dim_text
        # codet5 classification
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.fc = nn.Linear(self.dim_text, 1)
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.norm = nn.LayerNorm(self.dim_text)
        # self.bnorm = nn.BatchNorm1d(self.dim_text)
    
    def forward(
            self,
            text1,
            mask1,
            label1 = None,
            training_classifier=True,
    ):
        # training a classifier
        if training_classifier:
            enc_text = self.model(text1, attention_mask=mask1).last_hidden_state
            mask1_sum = mask1.sum(dim=-1, keepdims=True)
            enc_text = enc_text.masked_fill(mask1.unsqueeze(-1)==0, 0).sum(dim=1)/mask1_sum
            # outputs
            outputs = self.fc(enc_text)
            # loss
            if label1!=None:
                loss = self.loss(outputs.view(-1,1), label1.view(-1,1).to(torch.float16))
            else:
                loss = None
            preds = torch.sigmoid(outputs.view(-1,1)) >= 0.5
            preds = torch.squeeze(preds).float()
            return preds, loss


import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                  2))
import torch.nn.init as init
class CodeT5ClassCA(torch.nn.Module):
    def __init__(self, args, model_name, dim_text=768):
        super(CodeT5ClassCA, self).__init__()
        print("You are training with CodeT5ClassCA model.")
        self.args = args
        # dim_text = 768
        # codet5 classification
        self.codeT5 = T5EncoderModel.from_pretrained(model_name, trust_remote_code=True)
        # self.to_text_latent = nn.Linear(dim_text, dim_text)
        # self.to_text_latent2 = nn.Linear(dim_text, dim_text)
        self.cross_att = CrossAttention(dim_text)
        self.fc = nn.Linear(dim_text*2, 1)
        self.loss = torch.nn.BCEWithLogitsLoss() 
        if 'contrastive' in args.output_dir.lower() or 'loadcl' in args.output_dir.lower():
            self.dense = nn.Linear(dim_text*2, 2)
        else:
            self.dense = nn.Linear(dim_text, 2)
        self.c_loss = ContrastiveLoss()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # init.xavier_uniform_(self.to_text_latent.weight)
        # init.xavier_uniform_(self.to_text_latent2.weight)
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.dense.weight)

    def forward(
            self,
            text1=None,
            mask1=None,
            text2=None,
            mask2=None,
            text3=None,
            mask3=None,
            label1 = None,
            label2 = None,
            training_classifier=True,
    ):
        # training a classifier
        if training_classifier:
            enc_text1 = self.codeT5(text1, attention_mask=mask1).last_hidden_state #nl [b, dim]
            enc_text2 = self.codeT5(text2, attention_mask=mask2).last_hidden_state #code2 [b, dim]
            
            # x1 = enc_text1.mean(dim=1)
            x1 = enc_text1.mean(dim=1) #self.to_text_latent(enc_text1.mean(dim=1))
            x2 = enc_text2.mean(dim=1) # self.to_text_latent2(enc_text2.mean(dim=1))

            # outputs
            outputs = self.fc(torch.cat([x1, x2],dim=-1))
            # loss
            if label1!=None:
                loss = self.loss(outputs, label1.view(-1,1).to(torch.float16))
            else:
                loss = None
            preds = torch.sigmoid(outputs.view(-1,1)) >= 0.5
            preds = torch.squeeze(preds).float()

            return preds, loss
        else:
            # pass
            enc_text1 = self.codeT5(text1, attention_mask=mask1).last_hidden_state
            enc_text2 = self.codeT5(text2, attention_mask=mask2).last_hidden_state
            enc_text3 = self.codeT5(text3, attention_mask=mask3).last_hidden_state
            x1 = torch.cat([enc_text1.mean(dim=1), enc_text2.mean(dim=1)], dim=-1)
            x2 = torch.cat([enc_text1.mean(dim=1), enc_text3.mean(dim=1)], dim=-1)
            # x1 = enc_text2.mean(dim=1)
            # x2 = enc_text3.mean(dim=1)
            CLS1 = self.dense(x1)
            CLS2 = self.dense(x2)
            if label1!=None:
                loss = self.c_loss(CLS1, CLS2, label1)
            else:
                loss = None
            outputs = CLS1, CLS2
            return outputs, loss
            

from transformers import AutoTokenizer, AutoModelForSequenceClassification
class DsClassCA(torch.nn.Module):
    def __init__(self, args, model_name, dim_text=768):
        super(DsClassCA, self).__init__()
        print("You are training with DsClassCA model.")
        self.args = args
        # codet5 classification
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # self.to_text_latent = nn.Linear(dim_text, dim_text)
        self.cross_att = CrossAttention(dim_text)
        self.fc = nn.Linear(dim_text*2, 1)
        self.loss = torch.nn.BCEWithLogitsLoss() 
        self.dense = nn.Linear(dim_text*2, 2)
        # self.dense = nn.Linear(dim_text, 2)
        # self.norm = nn.LayerNorm(dim_text)
        self.c_loss = ContrastiveLoss()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # init.xavier_uniform_(self.to_text_latent.weight)
        # init.xavier_uniform_(self.to_text_latent2.weight)
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.dense.weight)

    def forward(
            self,
            text1=None,
            mask1=None,
            text2=None,
            mask2=None,
            text3=None,
            mask3=None,
            label1 = None,
            label2 = None,
            training_classifier=True,
    ):
        # training a classifier
        if training_classifier:
            enc_text1 = self.model(text1, attention_mask=mask1).last_hidden_state #nl [b, seq, dim]
            enc_text2 = self.model(text2, attention_mask=mask2).last_hidden_state #code2 [b, seq, dim]
            enc_text1 = enc_text1.mean(dim=1)
            enc_text2 = enc_text2.mean(dim=1)

            outputs = self.fc(torch.cat([enc_text1, enc_text2],dim=-1))
            # loss
            if label1!=None:
                loss = self.loss(outputs, label1.view(-1,1).to(torch.float16))
            else:
                loss = None
            preds = torch.sigmoid(outputs.view(-1,1)) >= 0.5
            preds = torch.squeeze(preds).float()

            return preds, loss
        else:
            enc_text1 = self.model(text1, attention_mask=mask1).last_hidden_state
            enc_text2 = self.model(text2, attention_mask=mask2).last_hidden_state
            enc_text3 = self.model(text3, attention_mask=mask3).last_hidden_state
            x1 = torch.cat([enc_text1.mean(dim=1), enc_text2.mean(dim=1)], dim=-1)
            x2 = torch.cat([enc_text1.mean(dim=1), enc_text3.mean(dim=1)], dim=-1)
            # x1 = enc_text2.mean(dim=1)
            # x2 = enc_text3.mean(dim=1)
            CLS1 = self.dense(x1)
            CLS2 = self.dense(x2)
            if label1!=None:
                loss = self.c_loss(CLS1, CLS2, label1)
            else:
                loss = None
            outputs = CLS1, CLS2
            return outputs, loss



def build_or_load_gen_model(args, load_model=True):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # Loading from local
    local_dir = './Models/'
    print("Loading config from: ", local_dir+args.model_name_or_path)
    config = config_class.from_pretrained(local_dir+args.model_name_or_path, trust_remote_code=True)
    print("Loading tokenizer from: ", local_dir+args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(local_dir+args.model_name_or_path, trust_remote_code=True)
    print("Config: ", config)
    # args.dim_text = config.d_model
    if 'tag' in args.model_type or 'multitask' in args.model_type:
        print('Found tag in the model_type....')
        tokenizer.add_special_tokens({"additional_special_tokens": ['<CLASS_TAG>']})
        model = model_class(args, tokenizer, local_dir+args.model_name_or_path)
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("tokenizer.pad_token: ", tokenizer.pad_token)
        model = model_class(args, local_dir+args.model_name_or_path, config.hidden_size)
    logger.info(
        "Finish loading model [%s] from %s",
        get_model_size(model),
        args.model_name_or_path
    )
    return config, model, tokenizer

MODEL_CLASSES = {
    'codebert': (AutoConfig, CodeBERT, RobertaTokenizer),
    'codet5bce': (T5Config, CodeT5ClassBCE, RobertaTokenizer),
    'codet5ca': (AutoConfig, CodeT5ClassCA, AutoTokenizer),
    'deepseekca': (AutoConfig, DsClassCA, AutoTokenizer),
}
