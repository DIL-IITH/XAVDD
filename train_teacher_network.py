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
from tqdm import tqdm 
import time 
import numpy as np 
from utils import get_dataset,get_network,train_model,test_model,get_student_network
from prettytable import PrettyTable 
#parser 
ap = argparse.ArgumentParser()
ap.add_argument("--dataset",default='vgg10k',choices=['vgg','vgg10k','ave','music21'])
ap.add_argument("--device",default='cuda:0')
ap.add_argument("--batch_size",type=int,default=128)
ap.add_argument("--lr",type=float,default=0.001)
ap.add_argument("--epochs",type=int,default=100)
ap.add_argument("--model",default='convnet',choices=['convnet','resnet18'])
ap.add_argument("--fusion",default='ensemble',choices=['ensemble','concat'])
ap.add_argument("--with_cross_attn",action="store_true")
args = ap.parse_args()

#print the settings 
table = PrettyTable()
table.field_names = ["Description",'Parameter']
table.float_format = ".2"
table.add_row(["Model",args.model])
table.add_row(["Dataset",args.dataset])
table.add_row(["Batch Size",args.batch_size])
table.add_row(["Epochs",args.epochs])
table.add_row(["Fusion",args.fusion])
table.add_row(["cross attn",args.with_cross_attn])
print(table)


#global variables and folder creation
start_time = time.time() 
best_acc  = 0.0
test_acc  = 0.0
train_acc = 0.0
if not os.path.exists("pretrained/{}/{}".format(args.dataset,args.model)):
    os.makedirs("pretrained/{}/{}".format(args.dataset,args.model))

#obtain the dataset 
train_dataset,test_dataset,num_classes = get_dataset(args)
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers = 8,
    pin_memory=True,
    persistent_workers=True,
)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers = 2
)

#obtain the model
if args.with_cross_attn:
    model = get_network(args,num_classes).to(args.device)
    print("✅ BiXAttn model generated.")
else:
    model=get_student_network(args,num_classes).to(args.device)
    print("✅ Model w/o cross attention generated.")

#define the optimizer, criterion and learning rate scheduler
criterion = torch.nn.CrossEntropyLoss().to(args.device)
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
    eta_min=1e-5
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler,cosine_scheduler],
    milestones=[warmup_epochs]
)

#training loop
print(">>>> Starting the training loop....")
for epoch in range(1,args.epochs+1):
    train_loss,train_acc = train_model(
        model,
        trainloader,
        criterion,
        optimizer,
        args.device,
        epoch = epoch ,
        train_acc = train_acc 
    )

    if epoch%5==1 or epoch==args.epochs:
        test_acc = test_model(
            model,
            testloader,
            criterion,
            args.device
        )

        if test_acc > best_acc:
            best_epoch = epoch 
            tqdm.write("[UPDATE] Epoch: {} Test accuracy improved from {:.3f} to {:.3f}".format(epoch,best_acc,test_acc))
            best_acc = test_acc 
            if args.with_cross_attn:
                torch.save(model.state_dict(),'pretrained/{}/{}/teacher_model_bixattn.pt'.format(args.dataset,args.model))
            else:
                torch.save(model.state_dict(),'pretrained/{}/{}/teacher_model_wo_attn.pt'.format(args.dataset,args.model))
    scheduler.step()

print("Total time taken for {} epochs : {:.3f} seconds".format(args.epochs,time.time()-start_time))
print("Best test accuracy achieved: {:.3f} at epoch: {}".format(best_acc,best_epoch))
