from utils import get_dataset_classwise
from tqdm import tqdm 
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch 
import argparse 
import os 

ap=argparse.ArgumentParser()
ap.add_argument("--dataset",required=True)
args=ap.parse_args()

if args.dataset=='vgg10k':
    num_classes = 10 
elif args.dataset == 'music21':
    num_classes = 21 
elif args.dataset == 'ave':
    num_classes = 28 
elif args.dataset == 'vgg':
    num_classes = 309 

mean = [0.425, 0.396, 0.370]
std =  [0.229, 0.224, 0.221]
normalize = transforms.Normalize(mean,std)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize])

if not os.path.exists('init_data/{}/audio'.format(args.dataset)):
    os.makedirs('init_data/{}/audio'.format(args.dataset))

if not os.path.exists('init_data/{}/frame'.format(args.dataset)):
    os.makedirs('init_data/{}/frame'.format(args.dataset))  

for class_id in tqdm(range(num_classes)):
    train_dataset =get_dataset_classwise(args,class_id)
    trainloader = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=False
    )

    train_iterator = iter(trainloader)
    img,aud=next(train_iterator)
    torch.save(img.detach().clone(),'init_data/{}/frame/tensor_{}.pt'.format(args.dataset,class_id))
    torch.save(aud.detach().clone(),'init_data/{}/audio/tensor_{}.pt'.format(args.dataset,class_id))
    del trainloader , train_iterator

