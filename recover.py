#imports 
import torch 
import argparse 
import collections
import wandb 
import random 
import os 
import numpy as np 
from torchvision import transforms 
from tqdm import tqdm 
from utils import clip_data,denormalize,get_network,lr_cosine_policy
from utils import save_images_multiple
from torchmetrics.functional.pairwise import pairwise_cosine_similarity,pairwise_euclidean_distance
from prettytable import PrettyTable
#parser 
ap = argparse.ArgumentParser()
ap.add_argument("--iteration",type=int,default=1000)
ap.add_argument("--lr",type=float,default=0.1)
ap.add_argument("--device",default="cuda:0")
ap.add_argument("--attn_multiplier",type=float,default=0.1)
ap.add_argument("--jitter",default=4,type=int)
ap.add_argument("--cpc_start",type=int,default=0)
ap.add_argument("--cpc",type=int,default=10)
ap.add_argument("--start_class_id",type=int,default=0)
ap.add_argument("--batch_size",type=int,default=10)
ap.add_argument("--model",default="convnet",choices=['convnet','resnet18'])
ap.add_argument("--dataset",default='vgg10k',choices=['vgg10k','vgg','ave','music21'])
ap.add_argument("--arch_path",required=True)
ap.add_argument("--wandb",type=int,default=0)
args=ap.parse_args()

#print the settings 
table = PrettyTable()
table.field_names = ["Description",'Parameter']
table.float_format = ".2"
table.add_row(["Model",args.model])
table.add_row(["Dataset",args.dataset])
table.add_row(["Batch Size",args.batch_size])
table.add_row(["Iterations",args.iteration])
table.add_row(["arch path ",args.arch_path])
table.add_row(["Multiplier",args.attn_multiplier])
table.add_row(["Start class ID",args.start_class_id])
table.add_row(["cpc start",args.cpc_start])
table.add_row(["cpc",args.cpc])
print(table)

if args.wandb==1:
    wandb.init(
        project='XAVDD_distillation',
        name="{}_{}".format(args.dataset,args.model)
    )
if args.dataset == "vgg10k":
    num_classes = 10
elif args.dataset == "vgg":
    num_classes = 309
elif args.dataset == 'music21':
    num_classes = 21
elif args.dataset ==  'ave':
    num_classes = 28 
    

#load the model 
model = get_network(args,num_classes).to(args.device)
checkpoint = torch.load(args.arch_path,map_location=args.device)
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(new_state_dict)
model.eval()
for p in model.parameters():
    p.requires_grad = False 
print("✅ BiXAttn model generated.")

#setup and create folder to store synthetic dataset 
args.syn_data_path="syn_data/{}".format(args.dataset)
exp_name="{}/syn_av_init_attn".format(args.model)
args.syn_data_path = args.syn_data_path + '/'+exp_name 
if not os.path.exists(args.syn_data_path):
    os.makedirs(args.syn_data_path)

#real embeddings
audio_memory={}
attended_audio_memory={}
video_memory={}
attended_video_memory={}
original_length_dict={}

for class_id in range(args.start_class_id,args.start_class_id+args.batch_size):
    audio_memory[class_id]=torch.load('patch_embeddings/{}/{}/audio/class{}.pt'.format(args.dataset,args.model,class_id),map_location=args.device).detach().clone()
    attended_audio_memory[class_id]=torch.load('attended_embeddings/{}/{}/video/class{}.pt'.format(args.dataset,args.model,class_id),map_location=args.device).detach().clone()
    original_length_dict[class_id]=len(audio_memory[class_id])
    video_memory[class_id]=torch.load('patch_embeddings/{}/{}/video/class{}.pt'.format(args.dataset,args.model,class_id),map_location=args.device).detach().clone()
    attended_video_memory[class_id]=torch.load('attended_embeddings/{}/{}/audio/class{}.pt'.format(args.dataset,args.model,class_id),map_location=args.device).detach().clone()

best_cost = 1e4
prev_best_cost = 1e4
best_epoch = 0

targets = torch.LongTensor(np.repeat(np.arange(args.start_class_id,args.start_class_id+args.batch_size),args.cpc)).to(args.device)
for class_idx in range(args.start_class_id,args.start_class_id+args.batch_size):
    loaded_tensor_video = torch.load('init_data/{}/frame/tensor_{}.pt'.format(args.dataset,class_idx),map_location=args.device)

    if class_idx == args.start_class_id:
        inputs_video = loaded_tensor_video[args.cpc_start:args.cpc_start+args.cpc,:,:,:].detach().clone()
    else:
        inputs_video = torch.vstack((inputs_video,loaded_tensor_video[args.cpc_start:args.cpc_start+args.cpc,:,:,:].detach().clone()))
print(inputs_video.shape)
inputs_video.requires_grad_(True)

for class_idx in range(args.start_class_id,args.start_class_id+args.batch_size):
    loaded_tensor_audio = torch.load('init_data/{}/audio/tensor_{}.pt'.format(args.dataset,class_idx),map_location=args.device)
    if class_idx == args.start_class_id:
        inputs_audio = loaded_tensor_audio[args.cpc_start:args.cpc_start+args.cpc,:,:,:].detach().clone()
    else:
        inputs_audio = torch.vstack((inputs_audio,loaded_tensor_audio[args.cpc_start:args.cpc_start+args.cpc,:,:,:].detach().clone()))
print(inputs_audio.shape)
inputs_audio.requires_grad_(True)

iterations_per_layer = args.iteration 
lim_0, lim_1 = args.jitter, args.jitter
optimizer = torch.optim.Adam([inputs_audio,inputs_video], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(args.device)


pbar = tqdm(range(iterations_per_layer))
for iteration in range(iterations_per_layer):
    lr_scheduler(optimizer, iteration, iteration)
    aug_function = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])
    inputs_video_jit = aug_function(inputs_video)
    off1 = random.randint(0, lim_0)
    off2 = random.randint(0, lim_1)
    inputs_video_jit = torch.roll(inputs_video_jit, shifts=(off1, off2), dims=(2, 3))
    inputs_audio_jit = torch.roll(inputs_audio, shifts=(off1, off2), dims=(2, 3))
    optimizer.zero_grad()

    # with torch.no_grad():
    out_a,out_v=model(inputs_video_jit,inputs_audio_jit)

    for class_idx in range(args.start_class_id,args.start_class_id+args.batch_size):
        rand_indices = torch.randperm(original_length_dict[class_idx])
        real_audio_emb = audio_memory[class_idx][rand_indices[:args.cpc]]
        syn_attn_emb,_ = model.recover(inputs_video_jit[args.cpc*(class_idx-args.start_class_id):args.cpc*(class_idx-args.start_class_id+1)],real_audio_emb,syn_modality='video')
        real_attn_emb = attended_audio_memory[class_idx]
        euc_sim = pairwise_euclidean_distance(real_attn_emb, syn_attn_emb, zero_diagonal=True)
        cos_sim = pairwise_cosine_similarity(syn_attn_emb,zero_diagonal=True)
        if class_idx == args.start_class_id:
            attn_loss_audio = torch.mean(torch.abs(euc_sim))
            syn_loss_audio = torch.mean(torch.abs(cos_sim))
        else:
            attn_loss_audio += torch.mean(torch.abs(euc_sim))
            syn_loss_audio += torch.mean(torch.abs(cos_sim))

    for class_idx in range(args.start_class_id,args.start_class_id+args.batch_size):
        rand_indices = torch.randperm(original_length_dict[class_idx])
        real_video_emb = video_memory[class_idx][rand_indices[:args.cpc]]
        _,syn_attn_emb = model.recover(real_video_emb,inputs_audio_jit[args.cpc*(class_idx-args.start_class_id):args.cpc*(class_idx-args.start_class_id+1)],syn_modality='audio')
        real_attn_emb = attended_video_memory[class_idx]
        euc_sim = pairwise_euclidean_distance(real_attn_emb, syn_attn_emb, zero_diagonal=True)
        if class_idx == args.start_class_id:
            attn_loss_video = torch.mean(torch.abs(euc_sim))
            syn_loss_video = torch.mean(torch.abs(cos_sim))
        else:
            attn_loss_video += torch.mean(torch.abs(euc_sim))
            syn_loss_video += torch.mean(torch.abs(cos_sim))

    optimizer.zero_grad()
    loss_ce_a =  criterion(out_a, targets)
    loss_ce_v = criterion(out_v,targets)



    loss = loss_ce_a + loss_ce_v + args.attn_multiplier*(attn_loss_audio+attn_loss_video)
    metrics = {
        'loss_ce_a':loss_ce_a,
        'loss_ce_v':loss_ce_v,
        'audio':args.attn_multiplier*attn_loss_audio,
        'video':args.attn_multiplier*attn_loss_video,
        'total':loss.item()
    }
    if args.wandb==1:
        wandb.log(metrics)
    loss.backward()
    optimizer.step()

    inputs_video.data = clip_data(inputs_video.data)

    if (iteration%20==1 and best_cost > loss.item()) or iteration == 0:
        prev_best_cost = best_cost 
        best_inputs_audio = inputs_audio.data.clone()
        save_images_multiple(args,best_inputs_audio,targets,args.cpc,for_audio=1)
        best_inputs_video = denormalize(inputs_video.data.clone())
        save_images_multiple(args,best_inputs_video,targets,args.cpc,for_audio=0)
        best_cost = loss.item()
        best_epoch = iteration 
        pbar.set_postfix_str("[{}]Cost improved from {:.5f} to {:.5f}".format(best_epoch,prev_best_cost,best_cost))
    pbar.update(1)
optimizer.state = collections.defaultdict(dict)
torch.cuda.empty_cache()
    