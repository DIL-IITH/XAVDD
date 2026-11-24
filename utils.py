#imports 
import torch 
from torch.utils.data import Dataset,DataLoader 
from torchaudio import transforms as T 
import torchvision 
from tqdm import tqdm 
from PIL import Image 
import torchvision.transforms.autoaugment as autoaugment
from torchvision import transforms 
from torch.nn import functional as F 
import glob 
import numpy as np 
import os 
#Dataset definition 
class DatasetClasswise(Dataset):
    def __init__(self,data_path,transforms=None,class_id=0):
        self.audio_path = data_path+'/audio/class{}'.format(class_id) 
        self.video_path = data_path+'/video/class{}'.format(class_id)
        self.transform  = transforms
        self.data       = []

        no_of_files = len(glob.glob(self.video_path+'/*.pt'))
        for i in range(no_of_files):
            audio_path=self.audio_path+'/tensor{}.pt'.format(i)
            video_path=self.video_path+'/tensor{}.pt'.format(i)
            self.data.append([audio_path,video_path])
        print("✅ Loaded from data path: ",data_path)
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        audio_path,img_path=self.data[idx]
        aud = torch.load(audio_path,map_location='cpu')
        img = torch.load(img_path,map_location='cpu')
        if self.transform is not None:
            img = self.transform(img)
        return img,aud

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        # open lazily to avoid multiprocess issues
        self.file = None

        # read length once
        with h5py.File(h5_path, 'r') as f:
            self.length = f['labels'].shape[0]

    def __len__(self):
        return self.length

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        img = torch.from_numpy(self.file['images'][idx])
        if self.transform:
            img = self.transform(img)
        lbl = torch.tensor(self.file['labels'][idx], dtype=torch.long)
        aud = torch.from_numpy(self.file['audio'][idx])
        
        augmenter=T.FrequencyMasking(freq_mask_param=15)
        if torch.rand(1)>0.5:
            aud = augmenter(aud)
        augmenter=T.TimeMasking(time_mask_param=10)
        if torch.rand(1)>0.5:
            aud = augmenter(aud)
        
        augmenter = torchvision.transforms.RandomHorizontalFlip()
        if torch.rand(1)>0.5:
            aud = augmenter(aud)
        
        return img,aud,lbl 

class HDF5DatasetClasswise(torch.utils.data.Dataset):
    def __init__(self, h5_path, transforms=None, class_id=0):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied on a sample.
            modality (str): 'video' or 'audio'.
            class_filter (int, optional): If provided, only samples with this label are included.
        """
        self.h5_path = h5_path
        self.transform = transforms
        self.file = None
        self.class_filter = class_id 

        # read indices once
        with h5py.File(h5_path, 'r') as f:
            all_labels = f['labels'][:]
            if self.class_filter is not None:
                self.indices = [i for i, lbl in enumerate(all_labels) if lbl == self.class_filter]
            else:
                self.indices = list(range(len(all_labels)))

        self.length = len(self.indices)

    def __len__(self):
        return self.length

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = self.indices[idx]
        lbl = torch.tensor(self.file['labels'][real_idx], dtype=torch.long)


        img = torch.from_numpy(self.file['images'][real_idx])
        if self.transform:
            img = self.transform(img)
        aud = torch.from_numpy(self.file['audio'][real_idx])
        return img,aud
#obtain syn dataset 
class SynDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,transforms=None,ipc=10):
        self.audio_path = data_path
        self.video_path = data_path 
        self.transform  = transforms
        self.data       = []
        self.class_map  = {}


        self.video_folder_list=glob.glob(self.video_path+'/*')
        for i in range(len(self.video_folder_list)):
            self.class_map['new{:03d}'.format(i)]=i 
        
        for class_path in self.video_folder_list:
            class_name = class_path.split("/")[-1]
            no_of_files = len(glob.glob(class_path+"/*.png"))
            list_of_files = list(range(no_of_files))
            for i1 in range(ipc):
                i=list_of_files[i1]
                j=list_of_files[i1]
                img_path_syn = self.video_path + class_name +"/class{:03d}_id{:03d}.png".format(self.class_map[class_name],i)
                aud_path_syn = self.audio_path + class_name +"/class{:03d}_id{:03d}.pt".format(self.class_map[class_name],j)
                self.data.append([img_path_syn,aud_path_syn,self.class_map[class_name]])
        
        print("[INFO] Loaded {} samples from {}".format(len(self.data),data_path))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        img_path,audio_path,labels = self.data[idx]
        aud = torch.load(audio_path,map_location='cpu')

        augmenter=T.FrequencyMasking(freq_mask_param=15)
        aud = augmenter(aud)
        augmenter=T.TimeMasking(time_mask_param=10)
        aud = augmenter(aud)
        augmenter = torchvision.transforms.RandomHorizontalFlip()
        aud = augmenter(aud)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img,aud,labels

#convnet audio model
class ConvNet_a(torch.nn.Module):
    def __init__(self,channel=1,im_size=(128,56)):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            # (128, 56) → (64, 28)
            torch.nn.Conv2d(channel,32,kernel_size=3,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            # (64, 28) → (32, 14)
            torch.nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            # (32, 14) → (16, 7)
            torch.nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            # (16, 7) → (7, 7)
            torch.nn.AdaptiveAvgPool2d((7,7))
        )
        self.embed_dim = 128

    def forward(self,x,return_tokens=True):
        feat = self.backbone(x)
        if not return_tokens:
            return feat 
        B,C,H,W = feat.shape
        tokens = feat.flatten(2).transpose(1,2)
        return tokens 


#convnet video model
class ConvNet_v(torch.nn.Module):
    def __init__(self,channel=1,im_size=(224,224)):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            # (224, 224) → (112, 112)
            torch.nn.Conv2d(channel,32,kernel_size=3,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            # (112, 112) → (56, 56)
            torch.nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            # (56, 56) → (28, 28)
            torch.nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            # (28, 28) → (14, 14)
            torch.nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            # (14, 14) → (7, 7)
            torch.nn.AdaptiveAvgPool2d((7,7))
        )
        self.embed_dim = 128

    def forward(self,x,return_tokens=True):
        feat = self.backbone(x)
        if not return_tokens:
            return feat 
        B,C,H,W = feat.shape
        tokens = feat.flatten(2).transpose(1,2)
        return tokens 

class CrossAttentionLayer(torch.nn.Module):

    def __init__(self,embed_dim,num_heads,dropout = 0.1):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True 
        )
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,query,key_val):
        attn_out,_ = self.attn(query,key_val,key_val)
        return self.norm(query + self.dropout(attn_out))

# bi-directional cross-attention block 
class BiDirectionalCrossAttentionBlock(torch.nn.Module):
    def __init__(self,embed_dim,num_heads,ffn_ratio=4.0,dropout=0.1):
        super().__init__()
        ffn_dim = int(embed_dim * ffn_ratio)

        #cross attention in booth direction 
        self.audio_to_video = CrossAttentionLayer(embed_dim,num_heads,dropout)
        self.video_to_audio = CrossAttentionLayer(embed_dim,num_heads,dropout)

        #per-modality FFN 
        def make_ffn():
            return torch.nn.Sequential(
                torch.nn.Linear(embed_dim,ffn_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(ffn_dim,embed_dim),
                torch.nn.Dropout(dropout)
            )

        self.ffn_a = make_ffn()
        self.ffn_v = make_ffn()
        self.norm_a = torch.nn.LayerNorm(embed_dim)
        self.norm_v = torch.nn.LayerNorm(embed_dim)

    def forward(self,audio_tokens,video_tokens):
        a_star = self.audio_to_video(audio_tokens,video_tokens)
        v_star = self.video_to_audio(video_tokens,audio_tokens)

        #FFN with residual and linear norm 
        a_out = self.norm_a(a_star + self.ffn_a(a_star))
        v_out = self.norm_v(v_star + self.ffn_v(v_star))

        return a_out, v_out 



#combined AV models with bi-directional cross attention 
class CombinedAVModel(torch.nn.Module):
    def __init__(self,num_classes,embed_dim,model_type='convnet'):
        super().__init__()

        if model_type=='convnet':
            self.audio_encoder = ConvNet_a(channel=1,im_size=(128,56))
            self.video_encoder = ConvNet_v(channel=3,im_size=(224,224))

        self.embed_dim = embed_dim
        self.classifier_a = torch.nn.Linear(embed_dim,num_classes)
        self.classifier_v = torch.nn.Linear(embed_dim,num_classes)     

    def forward(self,img,aud):
        #extract embeddings 
        a_emb = self.audio_encoder(aud,return_tokens=False).reshape(-1,self.embed_dim)
        v_emb = self.video_encoder(img,return_tokens=False).reshape(-1,self.embed_dim)

        #classify each modality independently
        out_a = self.classifier_a(a_emb)
        out_v = self.classifier_v(v_emb)

    
        return out_a,out_v 


#class definition for VGGDataset 
class VGGDataset(Dataset):
    def __init__(self,data_path,transforms=None,is_test_data=False):
        self.audio_path         = data_path + '/audio'
        self.video_path         = data_path + '/video'
        self.transform          = transforms 
        self.data               = []
        self.is_test_data       = is_test_data 
        self.audio_folder_list  = glob.glob(self.audio_path+'/*')
        self.video_folder_list  = glob.glob(self.video_path+'/*')
        self.classes            = list(range(len(self.audio_folder_list)))

        for class_id in self.classes:
            no_of_files = len(glob.glob(self.audio_path+'/class{}/*.pt'.format(class_id)))
            if self.is_test_data:
                for i in range(no_of_files):
                    audio_path=self.audio_path+'/class{}/tensor{}.pt'.format(class_id,i)
                    video_path=self.video_path+'/class{}/tensor{}.pt'.format(class_id,i)
                    self.data.append([audio_path,video_path,class_id])
            else:
                for i in range(no_of_files):
                    audio_path=self.audio_path+'/class{}/tensor{}.pt'.format(class_id,i)
                    video_path=self.video_path+'/class{}/tensor{}.pt'.format(class_id,i)
                    self.data.append([audio_path,video_path,class_id])
        print("✅ Loaded {} samples from {} classes of  dataset from {}.".format(len(self.data),len(self.classes),data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        audio_path,img_path,labels = self.data[idx]
        img = torch.load(img_path,map_location='cpu')
        aud = torch.load(audio_path,map_location='cpu')
        if self.transform is not None:
            img = self.transform(img)
        if not self.is_test_data:
 
            augmenter=T.FrequencyMasking(freq_mask_param=15)
            if torch.rand(1)>0.5:
                aud = augmenter(aud)
            augmenter=T.TimeMasking(time_mask_param=10)
            if torch.rand(1)>0.5:
                aud = augmenter(aud)
            augmenter = torchvision.transforms.RandomHorizontalFlip()
            if torch.rand(1)>0.5:
               aud = augmenter(aud)
        return img,aud,labels 

class Music21Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,transforms=None,is_test_data=False):
        self.audio_path = data_path+'/audio' 
        self.video_path = data_path + '/video'
        self.transform  = transforms
        self.data       = []
        self.is_test_data=is_test_data

        self.audio_folder_list=glob.glob(self.audio_path+'/*')
        self.video_folder_list=glob.glob(self.video_path+'/*')
        self.classes =list(range(len(self.audio_folder_list)))
        for class_id in self.classes:
            no_of_files = len(glob.glob(self.audio_path+'/class{}/*.pt'.format(class_id)))
            if self.is_test_data:
                for i in range(no_of_files):
                    audio_path=self.audio_path+'/class{}/tensor_{}.pt'.format(class_id,i)
                    video_path=self.video_path+'/class{}/tensor_{}.pt'.format(class_id,i)
                    self.data.append([audio_path,video_path,class_id])
            else:
                for i in range(no_of_files):
                    audio_path=self.audio_path+'/class{}/tensor{}.pt'.format(class_id,i)
                    video_path=self.video_path+'/class{}/tensor{}.pt'.format(class_id,i)
                    self.data.append([audio_path,video_path,class_id])
        print("✅ Loaded {} samples from {} classes of  dataset from {}.".format(len(self.data),len(self.classes),data_path))
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        audio_path,img_path,labels = self.data[idx]
        img = torch.load(img_path,map_location='cpu')
        aud = torch.load(audio_path,map_location='cpu')
        if self.transform is not None:
            img = self.transform(img)
        if not self.is_test_data:
            augmenter=T.FrequencyMasking(freq_mask_param=15)
            aud = augmenter(aud)
            augmenter=T.TimeMasking(time_mask_param=10)
            aud = augmenter(aud)
            augmenter = torchvision.transforms.RandomHorizontalFlip()
            aud = augmenter(aud)
        return img,aud,labels 

def get_dataset(args):
    #preapration of datasets
    mean = [0.425, 0.396, 0.370]
    std =  [0.229, 0.224, 0.221]
    normalize = transforms.Normalize(mean,std)

    test_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img * (1/255.0)),
        normalize
    ])

    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img * (1/255.0)),
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        normalize])

    if args.dataset=='vgg10k':
        train_dataset = VGGDataset(
            data_path='data/VGG10K/train_data',
            transforms=train_transforms,
            is_test_data=False
        )
        test_dataset = VGGDataset(
            data_path='data/VGG10K/test_data',
            transforms=test_transforms,
            is_test_data=True
        )
        num_classes = 10
    elif args.dataset == 'music21':
        train_dataset = Music21Dataset(
            data_path='data/Music_21/music_21_train',
            transforms = train_transforms,
            is_test_data=False 
        )
        test_dataset = Music21Dataset(
            data_path='data/Music_21/music_21_test',
            transforms = test_transforms,
            is_test_data=True 
        )
        num_classes = 21
    elif args.dataset == 'ave':
        train_dataset = HDF5Dataset("data/AVE/dataset.h5", transform=train_transforms)
        test_dataset = VGGDataset(
            data_path='data/AVE/ave_test',
            transforms = test_transforms,
            is_test_data=True 
        )
        num_classes = 28

    return train_dataset,test_dataset,num_classes 


def get_network(args,num_classes):
    
    if args.model == 'resnet18':
        embed_dim  = 512
        num_heads  = 8
        num_layers = 2
    elif args.model == 'convnet':
        embed_dim = 128
        num_heads = 8
        num_layers = 2

    
    model = CombinedAVModelWithBiXAttn(
        num_classes = num_classes,
        embed_dim = embed_dim,
        num_heads = num_heads,
        num_layers = num_layers,
        dropout = 0.1,
        model_type = args.model
    )

    return model 

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model,trainloader,criterion,optimizer,device,epoch=0,train_acc=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(
        trainloader,
        desc=f"Epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0
    )

    for batch_idx, (img, aud, labels) in enumerate(pbar):

        img = img.to(device)
        aud = aud.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        out_a,out_v=model(img,aud)
        loss = criterion(out_a,labels) + criterion(out_v,labels)
        preds_a = F.softmax(out_a,dim=1)
        preds_v = F.softmax(out_v,dim=1)
        preds = (preds_a+preds_v)/2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _,predicted = torch.max(preds.data,1)
        total  += labels.size(0)
        correct += (predicted==labels).sum().item()
        pbar.set_postfix(
            loss=f"{running_loss/(batch_idx+1):.4f}",
            acc=f"{correct*100.0/total:.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}"
        )


    return running_loss,correct*100.0/total 


    
def test_model(model,testloader,criterion,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img,aud,labels in testloader:
            img,aud,labels = img.to(device),aud.to(device),labels.to(device)
            out_a,out_v=model(img,aud)
            preds_a = F.softmax(out_a, dim=1)
            preds_v = F.softmax(out_v, dim=1)
            preds = (preds_a + preds_v) / 2
            _,predicted = torch.max(preds.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct*100.0/total 


class CombinedAVModelWithBiXAttn(torch.nn.Module):
    def __init__(self,num_classes,embed_dim,num_heads,num_layers,dropout=0.1,model_type='convnet'):
        super().__init__()
        if model_type=='convnet':
            self.audio_encoder = ConvNet_a(channel=1,im_size=(128,56))
            self.video_encoder = ConvNet_v(channel=3,im_size=(224,224))


        #stacked bi-directional cross-attention blocks
        self.bixattn_layers = torch.nn.ModuleList([
            BiDirectionalCrossAttentionBlock(embed_dim,num_heads,dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classifier_a = torch.nn.Linear(embed_dim,num_classes)
        self.classifier_v = torch.nn.Linear(embed_dim,num_classes)

    @staticmethod 
    def _gap(tokens):
        return tokens.mean(dim=1)

    def predict(self,data,modality):
        if modality == 'audio':
            tokens = self.audio_encoder(data)
            return self.classifier_a(self._gap(tokens))
        else:
            tokens = self.video_encoder(data)
            return self.classifier_v(self._gap(tokens))        

    def forward(self,img,aud):
        #extract spatial tokens 
        a_tok = self.audio_encoder(aud)
        v_tok = self.video_encoder(img)

        #stack of bi-directional cross attention blocks 
        for layer in self.bixattn_layers:
            a_tok,v_tok = layer(a_tok,v_tok)

        #global average pool
        a_emb = self._gap(a_tok)
        v_emb = self._gap(v_tok)

        #classify each modality independently
        out_a = self.classifier_a(a_emb)
        out_v = self.classifier_v(v_emb)

        return out_a,out_v 

    def recover(self,img,aud,syn_modality):

        #extract spatial tokens 
        if syn_modality=='audio':
            a_tok = self.audio_encoder(aud)
            v_tok = img.clone()
        else:
            v_tok = self.video_encoder(img)
            a_tok = aud.clone()
        #stack of bi-directional cross attention blocks 
        for layer in self.bixattn_layers:
            a_tok,v_tok = layer(a_tok,v_tok)

        a_emb = self._gap(a_tok)
        v_emb = self._gap(v_tok)

        return a_emb,v_emb 

    def embeddings(self,img,aud):
        with torch.no_grad():
            audio_emb = self.audio_encoder(aud)
            video_emb = self.video_encoder(img)
            a_tok = audio_emb.clone()
            v_tok = video_emb.clone()

            for layer in self.bixattn_layers:
                a_tok,v_tok = layer(a_tok,v_tok)

            a_emb = self._gap(a_tok)
            v_emb = self._gap(v_tok)


        return audio_emb, video_emb,a_emb,v_emb 

def get_dataset_classwise(args,class_id):
    #preapration of datasets
    mean = [0.425, 0.396, 0.370]
    std =  [0.229, 0.224, 0.221]
    normalize = transforms.Normalize(mean,std)

    test_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img * (1/255.0)),
        normalize
    ])

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        normalize])

    if args.dataset=='vgg10k':
        train_dataset = DatasetClasswise(
            data_path='data/VGG10K/train_data/',
            transforms=train_transforms,
            class_id=class_id
        )
    elif args.dataset == 'music21':
        train_dataset = DatasetClasswise(
            data_path='data/Music_21/music_21_train/',
            transforms=train_transforms,
            class_id=class_id
        )
    elif args.dataset == 'ave':
        train_dataset = HDF5DatasetClasswise(
            h5_path='data/AVE/dataset.h5',
            transforms=train_transforms,
            class_id=class_id
        )

    return train_dataset 

def clip_data(image_tensor):
    """
    adjust the input based on mean and variance
    """
    mean = np.array([0.425, 0.396, 0.370])
    std = np.array([0.229, 0.224, 0.221])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def denormalize(image_tensor):
    """
    convert floats back to input, tiny-imagenet
    """
    mean = np.array([0.425, 0.396, 0.370])
    std = np.array([0.229, 0.224, 0.221])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def save_images_multiple(args,images,targets,images_per_class=1,for_audio=0):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()
        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        if for_audio==1:
            place_to_store = dir_path + '/class{:03d}_id{:03d}.pt'.format(class_id, args.cpc_start+(id%images_per_class))
        else:
            place_to_store = dir_path + '/class{:03d}_id{:03d}.png'.format(class_id, args.cpc_start+(id%images_per_class))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if for_audio==1:
            temp_file = images[id].detach().clone()
            torch.save(temp_file,place_to_store)
        else:
            image_uint8 = (
                images[id]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .numpy()
            )

            Image.fromarray(image_uint8).save(
                place_to_store,
                format="PNG",
            )



def get_student_network(args,num_classes):
    
    if args.model == 'resnet18':
        embed_dim  = 512
    elif args.model == 'convnet':
        embed_dim = 6272

    model = CombinedAVModel(
        num_classes = num_classes,
        embed_dim = embed_dim,
        model_type = args.model
    )

    return model 


def train_model_sl(
    model,
    teacher_network,
    trainloader,
    criterion_kl,
    optimizer,
    device,
    alpha=0.5,
    temperature=10.0):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = None 
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    dsa_strategy='color_crop_cutout_flip_scale_rotate'
    dsa_param=ParamDiffAug()
    for img,aud,labels in trainloader:
        img,aud,labels = img.float().to(device),aud.float().to(device),labels.to(device)
        aud = DiffAugment(aud, dsa_strategy, param=dsa_param)
        img = DiffAugment(img, dsa_strategy, param=dsa_param)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            teacher_out_a, teacher_out_v = teacher_network(img,aud)
            teacher_out_a = teacher_out_a.detach()
            teacher_out_v = teacher_out_v.detach()
            teacher_out_a_softmax = F.softmax(teacher_out_a/temperature,dim=1)
            teacher_out_v_softmax = F.softmax(teacher_out_v/temperature,dim=1)

            out_a,out_v = model(img,aud)
            out_a_log_softmax = F.log_softmax(out_a/temperature,dim=1)
            out_v_log_softmax = F.log_softmax(out_v/temperature,dim=1)

            loss = alpha*(criterion_ce(out_a,labels)+criterion_ce(out_v,labels))+(1-alpha)*(criterion_kl(out_a_log_softmax,teacher_out_a_softmax)*temperature**2 + criterion_kl(out_v_log_softmax,teacher_out_v_softmax)*temperature**2)
      
        preds_a = F.softmax(out_a,dim=1)
        preds_v = F.softmax(out_v,dim=1)
        preds = (preds_a+preds_v)/2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _,predicted = torch.max(preds.data,1)
        total  += labels.size(0)
        correct += (predicted==labels).sum().item()
    return running_loss,correct*100.0/total 

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}



def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


def get_syn_dataset(args):
    #preapration of datasets
    mean = [0.425, 0.396, 0.370]
    std =  [0.229, 0.224, 0.221]
    normalize = transforms.Normalize(mean,std)

    test_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img * (1/255.0)),
        normalize
    ])

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        autoaugment.RandAugment(num_ops=2, magnitude=9),
        torchvision.transforms.ToTensor(),
        normalize])

    if args.dataset=='vgg10k':
        train_dataset = SynDataset(
            data_path=args.syn_data_path,
            transforms=train_transforms,
            ipc=args.ipc
        )
        test_dataset = VGGDataset(
            data_path='data/VGG10K/test_data',
            transforms=test_transforms,
            is_test_data=True
        )
        num_classes = 10
    elif args.dataset=='music21':
        train_dataset = SynDataset(
            data_path=args.syn_data_path,
            transforms=train_transforms,
            ipc=args.ipc
        )
        test_dataset = Music21Dataset(
            data_path='data/Music_21/music_21_test',
            transforms = test_transforms,
            is_test_data=True 
        )
        num_classes = 21
    elif args.dataset=='ave':
        train_dataset = SynDataset(
            data_path=args.syn_data_path,
            transforms=train_transforms,
            ipc=args.ipc
        )
        test_dataset = VGGDataset(
            data_path='data/AVE/ave_test',
            transforms = test_transforms,
            is_test_data=True 
        )
        num_classes = 28
    return train_dataset,test_dataset,num_classes 
