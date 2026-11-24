#imports 
import argparse 
import torch 
import os 
from torchvision import transforms 
from utils import get_dataset_classwise,get_network

#parser 
ap = argparse.ArgumentParser()
ap.add_argument("--device",default="cuda:0")
ap.add_argument("--batch_size",type=int,default=1)
ap.add_argument("--arch_path",required=True)
ap.add_argument("--model",default="convnet",choices=['convnet','resnet18'])
ap.add_argument("--dataset",default="vgg10k",choices=['vgg10k','vgg','music21','ave'])
args = ap.parse_args()

#check if folders to store embeddings exist
#if not, create them 
if not os.path.exists('patch_embeddings/{}/{}/video'.format(args.dataset,args.model)):
    os.makedirs('patch_embeddings/{}/{}/video'.format(args.dataset,args.model))

if not os.path.exists('patch_embeddings/{}/{}/audio'.format(args.dataset,args.model)):
    os.makedirs('patch_embeddings/{}/{}/audio'.format(args.dataset,args.model))

if not os.path.exists('attended_embeddings/{}/{}/video'.format(args.dataset,args.model)):
    os.makedirs('attended_embeddings/{}/{}/video'.format(args.dataset,args.model))

if not os.path.exists('attended_embeddings/{}/{}/audio'.format(args.dataset,args.model)):
    os.makedirs('attended_embeddings/{}/{}/audio'.format(args.dataset,args.model))

if args.dataset=='vgg10k':
    num_classes = 10 
elif args.dataset == 'music21':
    num_classes = 21 
elif args.dataset == 'ave':
    num_classes = 28 
elif args.dataset == 'vgg':
    num_classes = 309 

#get the model and load the checkpoint 
model = get_network(args,num_classes).to(args.device)
print("✅ BiXAttn model for {} generated.".format(args.model))
checkpoint = torch.load(args.arch_path,map_location=args.device)
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(new_state_dict)
model.eval()
for p in model.parameters():
    p.requires_grad = False 

#class-wise obtain the embeddings 
for class_id in range(num_classes):
    print('---------- Class ID: {} ----------'.format(class_id))
    train_dataset = get_dataset_classwise(args,class_id)
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers = 2
    )
    counter =0
    for img,aud in trainloader:
        img,aud = img.to(args.device),aud.to(args.device)
        _audio_emb,_video_emb,_audio_attn,_video_attn=model.embeddings(img,aud)
        if counter==0:
            audio_emb = _audio_emb.detach().clone()
            video_emb = _video_emb.detach().clone()
            audio_attn = _audio_attn.detach().clone()
            video_attn = _video_attn.detach().clone()
            counter +=1 
        else:
            audio_emb = torch.vstack((audio_emb,_audio_emb.detach().clone()))
            video_emb = torch.vstack((video_emb,_video_emb.detach().clone()))
            audio_attn = torch.vstack((audio_attn,_audio_attn.detach().clone()))
            video_attn = torch.vstack((video_attn,_video_attn.detach().clone()))

    print(audio_emb.shape)
    print(video_emb.shape)
    print(audio_attn.shape)
    print(video_attn.shape)
    torch.save(audio_emb,'patch_embeddings/{}/{}/audio/class{}.pt'.format(args.dataset,args.model,class_id))
    torch.save(video_emb,'patch_embeddings/{}/{}/video/class{}.pt'.format(args.dataset,args.model,class_id))
    torch.save(audio_attn,'attended_embeddings/{}/{}/audio/class{}.pt'.format(args.dataset,args.model,class_id))
    torch.save(video_attn,'attended_embeddings/{}/{}/video/class{}.pt'.format(args.dataset,args.model,class_id))
