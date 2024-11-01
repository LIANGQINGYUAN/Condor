import argparse
import pprint
from models import build_or_load_gen_model
from data_utils import get_data, get_pos_data, CustomDataset, CLDataset
import os
import time
import math
from tqdm import tqdm
import json
from itertools import cycle

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, set_seed
from torch import cuda

import matplotlib.pyplot as plt

from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState

import wandb

def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)

def get_rank():
    return int(os.environ.get("RANK", "0"))
def get_world_size():
    return os.environ.get("CUDA_VISIBLE_DEVICES","0").count(',')+1

def validation(args, model, loader):
    model.eval()
    device = args.accelerator.device
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), desc="Eval: "):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            if 'codebert' in args.load:
                outputs, loss = model(ids, mask)
                fin_targets.extend(targets.cpu().detach().tolist())
                outputs = outputs.view(-1,1)
                tmp_outputs = torch.sigmoid(outputs).cpu().detach().tolist()
                tmp_outputs = np.array(tmp_outputs) >= 0.5
                fin_outputs.extend([1 if p[0]==True else 0 for p in tmp_outputs])
            else:
                outputs, loss = model(text1=ids, mask1=mask)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                # no sigmoid
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def validation_multigpu(args, model, loader, tokenizer=None):
    model.eval()
    loader = args.accelerator.prepare(loader)
    args.accelerator.print(f"The size of valid dataloader: {len(loader)}, {len(loader.dataset)}. (Batch size: {args.batch_size}, Word size: {torch.cuda.device_count()})")
    device = args.accelerator.device
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        index = 0
        for data in tqdm(loader, total=len(loader), desc="Eval: "):
            ids = data['ids'].to(device, dtype = torch.long)
            # mask = data['mask'].to(device, dtype = torch.long)
            mask = ids.ne(tokenizer.pad_token_id)
            targets = data['targets'].to(device, dtype = torch.float)

            if 'codet5ca' in args.load or 'deepseekca' in args.load or 'codebert' in args.load:
                targets = data['targets'].to(device,  dtype = torch.long)
                ids2 = data['ids_2'].to(device, dtype = torch.long)
                # mask2 = data['mask_2'].to(device, dtype = torch.long)
                mask2 = ids2.ne(tokenizer.pad_token_id)
                outputs, loss = model(text1=ids, mask1=mask, label1=targets, text2=ids2, mask2=mask2)
                all_targets.extend(args.accelerator.gather_for_metrics(targets.cpu().detach().numpy().tolist()))
                all_predictions.extend(args.accelerator.gather_for_metrics(outputs.cpu().detach().numpy().tolist()))
            else:
                outputs, loss = model(text1=ids, mask1=mask)
                all_targets.extend(args.accelerator.gather_for_metrics(targets.cpu().detach().numpy().tolist()))
                all_predictions.extend(args.accelerator.gather_for_metrics(outputs.cpu().detach().numpy().tolist()))
    
    all_predictions = all_predictions[:len(loader.dataset)]
    all_targets = all_targets[:len(loader.dataset)]
    
    return all_predictions, all_targets

def run(args, model, tokenizer, optimizer, data_list):
    t0 = time.time()
    if args.accelerator.is_main_process:
        f_summary = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    training_loader, validation_loader, testing_loader = data_list
    device = args.accelerator.device
    args.do_train = True
    training_classifier = args.train_classifier
    print(f'Training config: is traing {args.do_train}, is validating {args.do_eval}, \
            is testing {args.do_test}, is training classifier {training_classifier}, load model {args.load}\
                length of training_loader {len(training_loader)}')
    if args.do_train:
        best_accf1 = 0   
        for epoch in range(args.epochs):
            model.train()
            if not training_classifier:
                plot_data = []
                plot_data_2 = []
                plot_label = []
                plot_data_nl = []
                plot_data_code2 = []

                dis_pos_record_list = []
                dis_neg_record_list = []

            bar = tqdm(training_loader, total=len(training_loader), desc="Training")
            for idx, data in enumerate(bar):
                ids = data['ids'].to(device, dtype = torch.long)
                # mask = data['mask'].to(device, dtype = torch.long)
                mask = ids.ne(tokenizer.pad_token_id)
                if 'codet5ca' in args.load or 'deepseekca' in args.load or 'codebert' in args.load:
                    targets = data['targets'].to(device,  dtype = torch.long)
                    if 'contrastive' in args.output_dir.lower():
                        ids2 = data['ids_code1'].to(device, dtype = torch.long)
                        ids3 = data['ids_code2'].to(device, dtype = torch.long)
                        mask2 = ids2.ne(tokenizer.pad_token_id)
                        mask3 = ids3.ne(tokenizer.pad_token_id)
                        outputs, loss = model(text1=ids, mask1=mask, label1=targets, text2=ids2, mask2=mask2, text3=ids3, mask3=mask3, training_classifier=training_classifier)
                    else:
                        ids2 = data['ids_2'].to(device, dtype = torch.long)
                        mask2 = ids2.ne(tokenizer.pad_token_id)
                        outputs, loss = model(text1=ids, mask1=mask, label1=targets, text2=ids2, mask2=mask2, training_classifier=training_classifier)
                else:
                    # print('training normal')
                    targets = data['targets'].to(device,  dtype = torch.long)
                    outputs, loss = model(text1=ids, mask1=mask, label1=targets, training_classifier=training_classifier)
                    
                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                if (idx+1) % args.grad_acc_steps == 0:
                    optimizer.zero_grad()
                    # loss.sum().backward()
                    args.accelerator.backward(loss)
                    optimizer.step()
                    if args.accelerator.is_main_process:
                        bar.set_description("[{}] Step: {}/{} Train loss {}".format(epoch+1, idx+1, len(training_loader), loss*args.grad_acc_steps))
                else:
                    with args.accelerator.no_sync(model):
                        args.accelerator.backward(loss)

                if not training_classifier:
                    # print("outputs: ", outputs[1])
                    # plot_data.extend(args.accelerator.gather_for_metrics(outputs[1].cpu().detach().numpy().tolist()))
                    plot_data.extend(args.accelerator.gather_for_metrics(outputs[0].cpu().detach().numpy().tolist()))
                    plot_label.extend(args.accelerator.gather_for_metrics(targets.cpu().detach().numpy().tolist()))

                    plot_data_2.extend(args.accelerator.gather_for_metrics(outputs[1].cpu().detach().numpy().tolist()))

            if args.accelerator.is_main_process and not training_classifier:
                plot_data = torch.tensor(plot_data[:len(training_loader.dataset)])
                plot_data_2 = torch.tensor(plot_data_2[:len(training_loader.dataset)])
                plot_label = plot_label[:len(training_loader.dataset)]
                # print("plot_data: ", plot_data.size(), plot_data)
                pic_path = os.path.join(args.output_dir,'checkpoint-pic')
                os.makedirs(pic_path, exist_ok=True)
                torch.save(plot_data, f'{pic_path}/{epoch}_data.pt')
                torch.save(plot_data_2, f'{pic_path}/{epoch}_data2.pt')
                torch.save(plot_label, f'{pic_path}/{epoch}_label.pt')
                plot_data = plot_data.to('cpu').detach().numpy()
                plt.scatter(plot_data[:, 0], plot_data[:, 1], c=plot_label, s=5, alpha=0.3)
                plt.savefig( os.path.join(pic_path, f'figure_{epoch}.png'))
                logger.info(f'figure_{epoch}.png')
                plt.close()

                save_path = os.path.join(args.output_dir,'checkpoint-cl')
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model.module if hasattr(model,'module') else model
                # model_to_save = args.accelerator.unwrap_model(model_to_save)
                torch.save(model_to_save.state_dict(), os.path.join(save_path, f'CLModel_{epoch}.bin'))
                logger.info("Save CL Model!")

            # args.do_eval = True
            if args.do_eval:
                # valid_outs, valid_tgts = validation(args, model, validation_loader)
                valid_outs, valid_tgts = validation_multigpu(args, model, validation_loader, tokenizer)
                print("Length of valid_outs: ", len(valid_outs))
                print("Length of valid_tgts: ", len(valid_tgts))
                if args.accelerator.is_main_process:
                    accuracy = metrics.accuracy_score(valid_tgts, valid_outs)
                    f1_score = metrics.f1_score(valid_tgts, valid_outs, average='weighted')
                    prediction_folder = os.path.join(args.output_dir,'prediction')
                    os.makedirs(prediction_folder, exist_ok=True)
                    with open(os.path.join(prediction_folder, f'valid_res_{epoch}.json'), 'w') as f:
                        for p, g in zip(valid_outs, valid_tgts):
                            f.write(json.dumps({"pred":p,"gold":g})+'\n')
                    accf1=accuracy+f1_score
                    logger.info(f'Accuracy:  {accuracy}, F1:  {f1_score}, Best Acc+F1:  {max(accf1, best_accf1)}')
                    wandb.log({"Accuracy": accuracy, "F1": f1_score, "loss": loss})
                    if accf1 > best_accf1:
                        not_inc_cnt = 0
                        f_summary.write("[%d] Best acc+f1 changed into %.2f (accuracy: %.2f, f1_score: %.2f)\n" % (epoch, accuracy+f1_score, accuracy, f1_score))
                        save_path = os.path.join(args.output_dir,'checkpoint-best')
                        os.makedirs(save_path, exist_ok=True)
                        best_accf1 = accuracy + f1_score
                        model_to_save = model.module if hasattr(model,'module') else model
                        # model_to_save = args.accelerator.unwrap_model(model_to_save)
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, 'model.bin'))
                        logger.info("Save Model!")
                    else:
                        not_inc_cnt += 1
                        f_summary.write("[%d] Best acc+f1 (%.2f) does not drop changed for %d epochs, cur acc+f1: %.2f (accuracy: %.2f, f1_score: %.2f)\n" % (
                                                epoch, best_accf1, not_inc_cnt, accuracy+f1_score, accuracy, f1_score))
                        
                        # save
                        save_path = os.path.join(args.output_dir,'checkpoins')
                        os.makedirs(save_path, exist_ok=True)
                        model_to_save = model.module if hasattr(model,'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, f'model_{epoch}.bin'))

                        if not_inc_cnt>=3:
                            stop_early_str = "[%d] Early stop as not_inc_cnt=%d\n" % (epoch, not_inc_cnt)
                            logger.info(stop_early_str)
                            args.accelerator.set_trigger()
                if args.accelerator.check_trigger():
                    break

    print("Start testing.......")
    # args.do_test = True
    if args.do_test:
        save_path = os.path.join(args.output_dir,'checkpoint-best')
        prediction_folder = os.path.join(args.output_dir,'prediction')
        model = model.module if hasattr(model,'module') else model
        # model = args.accelerator.unwrap_model(model)
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.bin'), map_location='cpu'))
        # outputs, targets = validation(args, model, testing_loader)
        outputs, targets = validation_multigpu(args, model, testing_loader, tokenizer)
        print("Length of outputs: ", len(outputs))
        print("Length of targets: ", len(targets))
        # outputs = np.array(outputs) >= 0.5
        if args.accelerator.is_main_process:
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score = metrics.f1_score(targets, outputs, average='weighted')
            logger.info(f"Accuracy Score = {accuracy}")
            logger.info(f"F1 Score = {f1_score}")
            with open(os.path.join(prediction_folder, 'test_res.json'), 'w') as f:
                for p, g in zip(outputs, targets):
                    # p = 1 if p[0]==True else 0
                    f.write(json.dumps({"pred":p,"gold":g})+'\n')
            logger.info("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.write("Testing results: Accuracy: {}, F1: {}".format(accuracy, f1_score))
            f_summary.write("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.close()

def main(args):
    accelerator = Accelerator()
    # mixed_precision = 'bf16'
    # args.mixed_precision = mixed_precision
    # accelerator = Accelerator(mixed_precision=mixed_precision)
    AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]=args.batch_size
    AcceleratorState().deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]=args.grad_acc_steps
    args.accelerator = accelerator

    # Model
    if 'codebert' in args.load:
        args.model_type = 'codebert'
        args.model_name_or_path = 'microsoft/codebert-base'
    elif 'deepseek' in args.output_dir.lower():
        args.model_type = args.load
        args.model_name_or_path = 'deepseek-coder-1.3b-base'
    elif 'codet5' == args.load:
        args.model_type = 'codet5'
        args.model_name_or_path = 'Salesforce/codet5-base'
    config, model, tokenizer = build_or_load_gen_model(args)
    logger.info("Load model from {}".format(args.model_name_or_path))
    accelerator.log(model)
    model = model.to(accelerator.device)

    # Config
    MAX_LEN = args.max_source_len
    TRAIN_BATCH_SIZE = args.batch_size
    VALID_BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    
    optimizer = torch.optim.Adam(params = model.parameters(), lr=args.lr)
    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Rank",
            # track hyperparameters and run metadata
            config=vars(args),
            name=args.task,
        )

    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, "drop_last":True}
    valid_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True}
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False}
    
    if 'codet5ca' in args.load or 'deepseekca' in args.load or'codebert' in args.load:
        if 'contrastive' in args.output_dir.lower():
            print("Prepare contrastive learning dataset......")
            training_set = CLDataset(args, tokenizer, MAX_LEN, data_type='train_pair_cl_sampled')
            validation_set = testing_set = None
        else:
            print("Prepare single with cross attention samples dataset......")
            training_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='train')
            validation_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='valid')
            testing_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='test')
    else:
        print("Prepare single samples dataset......")
        training_set = CustomDataset(args, tokenizer, MAX_LEN, data_type='train')
        validation_set = CustomDataset(args, tokenizer, MAX_LEN, data_type='valid')
        testing_set = CustomDataset(args, tokenizer, MAX_LEN, data_type='test')
        print("Length of training_set: ", len(training_set))

    training_loader = DataLoader(training_set, **train_params)
    if validation_set and testing_set:
        validation_loader = DataLoader(validation_set, **valid_params)
        testing_loader = DataLoader(testing_set, **test_params)
    else:
        validation_loader=None
        testing_loader=None


    if 'step2' in args.output_dir.lower() or 'contrastive' in args.output_dir.lower():
        args.train_classifier = False
        args.do_eval = False
        args.do_test = False
        logger.info("Futher training task......")
    else:
        if 'loadcl' in args.output_dir.lower():
            if 'codet5ca' in args.load:
                m_path = None # TODO: replace model weights
            else: # deepseek
                m_path = None 
            logger.info(f"Loading CL weights from {m_path} ......")
            model = model.module if hasattr(model,'module') else model
            model.load_state_dict(torch.load(m_path, map_location='cpu'))
        args.train_classifier = True
        args.do_eval = True
        args.do_test = True
        

    model = model.to(accelerator.device)
    data_list = [training_loader, validation_loader, testing_loader]

    if 'generation' in args.load.lower():
        logger.info("Generation task......")
        run_gen(args, model, tokenizer, optimizer, data_list)
    else:
        logger.info("Classification task......")
        run(args, model, tokenizer, optimizer, data_list)



def validation_multigpu_generation(args, model, loader, tokenizer=None):
    model.eval()
    loader = args.accelerator.prepare(loader)
    args.accelerator.print(f"The size of valid dataloader: {len(loader)}, {len(loader.dataset)}. (Batch size: {args.batch_size}, Word size: {torch.cuda.device_count()})")
    device = args.accelerator.device
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        index = 0
        eos_token_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
        for data in tqdm(loader, total=len(loader), desc="Eval: "):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = ids.ne(tokenizer.pad_token_id)
            targets = data['target_ids'].to(device, dtype = torch.long)
            targets_mask = targets.ne(tokenizer.pad_token_id)
            
            outputs = model.generate(ids, mask, max_length=args.max_source_len, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_token_id)
            all_targets.extend(args.accelerator.gather_for_metrics(targets.cpu().detach().numpy().tolist()))
            all_predictions.extend(args.accelerator.gather_for_metrics(outputs.cpu().detach().numpy().tolist()))
    all_predictions = all_predictions[:len(loader.dataset)]
    all_predictions = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in all_predictions]
    all_targets = all_targets[:len(loader.dataset)]
    all_targets = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in all_targets]
    return all_predictions, all_targets


def run_gen(args, model, tokenizer, optimizer, data_list):
    t0 = time.time()
    if args.accelerator.is_main_process:
        f_summary = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    training_loader, validation_loader, testing_loader = data_list
    device = args.accelerator.device
    args.do_train = True
    if args.do_train:
        best_accf1 = 0   
        for epoch in range(args.epochs):
            model.train()
            bar = tqdm(training_loader, total=len(training_loader), desc="Training")
            for idx, data in enumerate(bar):
                ids = data['input_ids'].to(device, dtype = torch.long)
                mask = ids.ne(tokenizer.pad_token_id)
                targets = data['labels'].to(device,  dtype = torch.long)
                targets_mask = targets.ne(tokenizer.pad_token_id)
                outputs, loss = model(text1=ids, mask1=mask, target_ids=targets)

                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                if (idx+1) % args.grad_acc_steps == 0:
                    optimizer.zero_grad()
                    args.accelerator.backward(loss)
                    optimizer.step()
                    if args.accelerator.is_main_process:
                        bar.set_description("[{}] Step: {}/{} Train loss {}".format(epoch+1, idx+1, len(training_loader), loss*args.grad_acc_steps))
                else:
                    with args.accelerator.no_sync(model):
                        args.accelerator.backward(loss)

            # args.do_eval = True
            if args.do_eval:
                valid_outs, valid_tgts = validation_multigpu_generation(args, model, validation_loader, tokenizer)
                print("Length of valid_outs: ", len(valid_outs))
                print("Length of valid_tgts: ", len(valid_tgts))
                if args.accelerator.is_main_process:
                    accuracy = sum([1 if p==g else 0 for p, g in zip(valid_outs, valid_tgts)])/len(valid_outs)
                    prediction_folder = os.path.join(args.output_dir,'prediction')
                    os.makedirs(prediction_folder, exist_ok=True)
                    with open(os.path.join(prediction_folder, f'valid_res_{epoch}.json'), 'w') as f:
                        for p, g in zip(valid_outs, valid_tgts):
                            f.write(json.dumps({"pred":p,"gold":g})+'\n')
                    accf1=accuracy
                    logger.info(f'Accuracy:  {accuracy}')
                    wandb.log({"Accuracy": accuracy, "loss": loss})
                    if accf1 >= best_accf1:
                        not_inc_cnt = 0
                        f_summary.write("[%d] Best acc changed into %.2f\n" % (epoch, accuracy))
                        save_path = os.path.join(args.output_dir,'checkpoint-best')
                        os.makedirs(save_path, exist_ok=True)
                        best_accf1 = accuracy
                        model_to_save = model.module if hasattr(model,'module') else model
                        # model_to_save = args.accelerator.unwrap_model(model_to_save)
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, 'model.bin'))
                        logger.info("Save Model!")
                    else:
                        not_inc_cnt += 1
                        f_summary.write(f"[{epoch}] Best acc+f1 ({best_accf1}) does not drop changed for {not_inc_cnt} epochs, cur acc: {accuracy}")
                        
                        # save
                        save_path = os.path.join(args.output_dir,'checkpoins')
                        os.makedirs(save_path, exist_ok=True)
                        model_to_save = model.module if hasattr(model,'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, f'model_{epoch}.bin'))

                        if not_inc_cnt>=3:
                            stop_early_str = "[%d] Early stop as not_inc_cnt=%d\n" % (epoch, not_inc_cnt)
                            logger.info(stop_early_str)
                            args.accelerator.set_trigger()
                if args.accelerator.check_trigger():
                    break

    print("Start testing.......")
    # args.do_test = True
    if args.do_test:
        save_path = os.path.join(args.output_dir,'checkpoint-best')
        prediction_folder = os.path.join(args.output_dir,'prediction')
        model = model.module if hasattr(model,'module') else model
        # model = args.accelerator.unwrap_model(model)
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.bin'), map_location='cpu'))
        # outputs, targets = validation(args, model, testing_loader)
        outputs, targets = validation_multigpu_generation(args, model, testing_loader, tokenizer)
        print("Length of outputs: ", len(outputs))
        print("Length of targets: ", len(targets))
        # outputs = np.array(outputs) >= 0.5
        if args.accelerator.is_main_process:
            accuracy = sum([1 if p==g else 0 for p, g in zip(valid_outs, valid_tgts)])/len(valid_outs)
            logger.info(f"Accuracy Score = {accuracy}")
            with open(os.path.join(prediction_folder, 'test_res.json'), 'w') as f:
                for p, g in zip(outputs, targets):
                    f.write(json.dumps({"pred":p,"gold":g})+'\n')
            logger.info("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.write(f"Testing results: Accuracy: {accuracy}")
            f_summary.write("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='Mutation', type=str)
    parser.add_argument('--data_dir', default='/raid/lqy/Mutation/Classifier/Dataset/Mutation', type=str)
    parser.add_argument('--cache_path', default='', type=str)
    parser.add_argument('--output_dir', default='./saved_models', type=str)

    parser.add_argument('--load', default='microsoft/codebert-base', type=str)
    parser.add_argument('--max_source_len', default=512, type=int)
    parser.add_argument('--max_target_len', default=-1, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--grad_acc_steps', default=1, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.cache_path = './cache/'
    os.makedirs(args.cache_path, exist_ok=True)
    argsdict = vars(args)
    with open(os.path.join(args.output_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))
    
    os.environ["http_proxy"] = "http://oversea-squid5.sgp.txyun:11080"
    os.environ["https_proxy"] = "http://oversea-squid5.sgp.txyun:11080"
    os.environ["WANDB_PROJECT"] = "Condor"
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_API_KEY"] = "fb84fe017f908c5ea8eb6062c177084ac614a86b"
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"


    set_seed(42)
    main(args)