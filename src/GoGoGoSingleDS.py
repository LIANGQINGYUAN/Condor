import os
import threading

# Set length
# src len, tgt len, 
task_length = { 
    "RealEdits": [512,3],
    "Mutation": [512,3],
    "OJ": [512,3],
}

def run(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc=5, LR=5e-5, EPOCH=10):
    os.system(f'bash ./run.sh "{port}" "{gpuid}" "{model}" "{model_tag}" "{task}" "{maxSL}" "{maxTL}" "{dataDir}" "{BSize}" "{GAcc}" "{LR}" "{EPOCH}"')

port = 8588


model = 'deepseekca'
tag = 'DeepSeek'
task='OJ'
maxSL, maxTL = task_length[task]
task='DeepSeekDatavOJAppsContrastive_baseline_Log' 
task=f'{task}_Scratch'
maxSL = 512
dataDir = './data_pair_cl'
BSize=15
GAcc=1
LR=5e-5
gpuid = '0,1,2,3,4,5,6,7'
port = port+1
EPOCH=50
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}'
# t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR, EPOCH)) 
# t.start()
run(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR, EPOCH)


model = 'deepseekca'
tag = 'DeepSeek'
task='OJ'
maxSL, maxTL = task_length[task]
task='DeepSeekDatavOJAppsMBPPCombine_baseline' 
task=f'{task}_Scratch'
maxSL = 512
dataDir = './dataset'
BSize=25
GAcc=1
LR=3e-6
gpuid = '0,1,2,3,4,5,6,7'
port = port+1
EPOCH=20
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}'
# t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR, EPOCH)) 
# t.start()
run(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR, EPOCH)


model = 'deepseekca'
tag = 'DeepSeek'
task='OJ'
maxSL, maxTL = task_length[task]
task='DeepSeekDatavOJAppsMBPPCombine_baseline_LoadCL' 
task=f'{task}_Scratch'
maxSL = 512
dataDir = './dataset'
BSize=25
GAcc=1
LR=3e-6
gpuid = '0,1,2,3,4,5,6,7'
port = port+1
EPOCH=20
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}'
# t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR, EPOCH)) 
# t.start()
run(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR, EPOCH)

