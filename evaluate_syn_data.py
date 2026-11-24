#imports and set environment variables
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import argparse 
import torchvision 
from tqdm import tqdm 
import numpy as np 
import datetime 
import csv 
import sys 
import time 
from utils import get_syn_dataset,get_network,get_student_network,train_model_sl,test_model 
from prettytable import PrettyTable

#parser 
ap=argparse.ArgumentParser()
ap.add_argument("--device",default="cuda:0")
ap.add_argument("--batch_size",type=int,default=128)
ap.add_argument("--lr",type=float,default=0.001)
ap.add_argument("--epochs",type=int,default=300)
ap.add_argument("--model",default="convnet",choices=['convnet','resnet18'])
ap.add_argument("--dataset",default='vgg10k',choices=['vgg10k','vgg','ave','music21'])
ap.add_argument("--syn_data_path",required=True)
ap.add_argument("--ipc",type=int,default=10)
ap.add_argument("--soft_label",type=int,default=1)
ap.add_argument("--iterations",type=int,default=5)
ap.add_argument("--alpha",type=float,default=0.5)
ap.add_argument("--temperature",type=int,default=10)
ap.add_argument("--type",required=True,choices=['baseline','ours'])
ap.add_argument("--arch_path",required=True)
args = ap.parse_args()

if args.model == "resnet18":
    embed_dim = 512 
elif args.model == "conv":
    embed_dim = 6272 
else:
    embed_dim = 192 

list_of_test_acc=[]
program_start_time=time.time()
if args.dataset == 'music21':
    test_workers = 8
else:
    test_workers =2 


for itr in range(args.iterations):
    print("=========== Iteration : {} / {} ==============".format(itr+1,args.iterations))
    table = PrettyTable()
    table.field_names = ["Description",'Parameter']
    table.float_format = ".2"
    table.add_row(["Model",args.model])
    table.add_row(["Syn data path",args.syn_data_path])
    table.add_row(["SPC",args.ipc])
    table.add_row(["Device",args.device])
    table.add_row(["Temperature",args.temperature])
    table.add_row(["Alpha",args.alpha])
    print(table)
    torch.manual_seed(itr)
    train_dataset,test_dataset,num_classes = get_syn_dataset(args)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=test_workers
    )

    #load the student model
    model = get_student_network(args,num_classes).to(args.device)
    if args.soft_label==1:
        teacher_model = get_student_network(args,num_classes).to(args.device)
        checkpoint = torch.load(args.arch_path,map_location=args.device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        teacher_model.load_state_dict(new_state_dict)
        teacher_model.eval()

        for p in teacher_model.parameters():
            p.requires_grad = False 


    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4,betas=(0.9,0.98),eps=1e-8)
    warmup_epochs = max(1,int(0.05*args.epochs))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor =1.0,
        total_iters = warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1,args.epochs-warmup_epochs),
        eta_min=1e-4
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler,cosine_scheduler],
        milestones=[warmup_epochs]
    )

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=False).to(args.device)

    #training loop
    pbar = tqdm(range(args.epochs),leave=True)
    best_acc = 0.0
    test_acc = 0.0
    for epoch in range(args.epochs):
        if args.soft_label==1:
            train_loss,train_acc=train_model_sl(model,teacher_model,trainloader,criterion_kl,optimizer,args.device,alpha=args.alpha,temperature= args.temperature)
        else:
            train_loss,train_acc = train_model(model,trainloader,criterion,optimizer,args.device)
        
        if (epoch%10==1 and epoch>150) or epoch==args.epochs-1:
            test_acc = test_model(model,testloader,criterion,args.device)
            if test_acc > best_acc:
                best_acc = test_acc 


        pbar.update(1)
        pbar.set_postfix_str("Train loss:{:.3f},train acc:{:.3f}, test acc:{:.3f}, lr: {:.4f}".format(train_loss,train_acc,best_acc,optimizer.param_groups[0]["lr"]))
        scheduler.step()
    print("🔵 Iteration : {}, Accuracy: {:.2f}".format(itr+1,best_acc))
    list_of_test_acc.append(best_acc)





print("==========================================")
print(list_of_test_acc)
print("Mean best acc:\t{:.3f}%".format(np.mean(list_of_test_acc)))
print("STD  best acc:\t{:.3f}%".format(np.std(list_of_test_acc)))

if not os.path.exists("progress_tracker"):
    os.mkdir("progress_tracker")

fields=['Timestamp','Model','IPC','soft_label','mean accuracy','std accuracy','Time','N','batch size','type','alpha','temp']


if args.soft_label==1:
    filename="progress_tracker/{}_data_distillation_sl_{}.csv".format(args.dataset,args.model)
else:
    filename="progress_tracker/{}_data_distillation_{}.csv".format(args.dataset,args.model)
if not os.path.exists(filename):
    with open(filename,"w") as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=fields)
        writer.writeheader()


result=[{
    'Timestamp':datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
    'Model':args.model,
    'soft_label':args.soft_label,
    'IPC':args.ipc,
    'mean accuracy':np.mean(list_of_test_acc),
    'std accuracy':np.std(list_of_test_acc),
    'Time':(time.time()-program_start_time)/args.iterations,
    'batch size':args.batch_size,
    'N':args.iterations,
    'type':args.type,
    'alpha':args.alpha,
    'temp':args.temperature
}]

with open(filename,'a+') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames=fields)
    writer.writerows(result)
