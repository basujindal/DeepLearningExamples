import os
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset

def load_imagenet_train_data(path, idx):
    data_dir = os.path.join(path , "train")
    train_data = np.load(os.path.join(data_dir, 'train_data_batch_' + str(idx) + '.npz'))
    return train_data['data'], train_data['labels']

class BatchDataImageNet(Dataset):

    def __init__(self, Path, imgTransform=None, num_images = None):
        self.path = Path

        self.img_trans = imgTransform

        self.allImgs, self.allLabels = load_imagenet_train_data(Path, idx)

        if num_images == None:
            num_images = len(self.imgs)
        for idx in tqdm(range(num_images)):

            if self.transform:
                image = self.image_trans(image)
                mask = self.transform(mask)
            
            self.allImgs[idx] = image.squeeze(0)
            self.allMasks[idx] = mask.squeeze(0)

        # self.final.to("cuda")

    train_data, train_labels = load_imagenet_train_data(1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.allImgs[idx], self.allMasks[idx]
    

class BatchDataImageMask(Dataset):

    def __init__(self, maskPath, imgPath, imgSize,imgC, imgTransform=None, maskExt="png", num_images = None):

        self.img_trans = imgTransform
        self.imgs = os.listdir(imgPath)
        self.masks = os.listdir(maskPath)
        self.maskExt = maskExt

        self.allImgs = torch.zeros([len(self.imgs),imgC, imgSize,imgSize], dtype=torch.float32)
        self.allMasks = torch.zeros([len(self.masks),1, imgSize,imgSize], dtype=torch.int8)

        if num_images == None:
            num_images = len(self.imgs)
        for idx in tqdm(range(num_images)):
            
            mask_path = os.path.join(maskPath, self.imgs[idx][:-3] + self.maskExt)
            mask = io.imread(mask_path)

            img_path = os.path.join(imgPath, self.imgs[idx])
            image = io.imread(img_path)
            # if len(mask.shape) < 3:
            #     image = np.expand_dims(image, axis = 2)

            if self.transform:
                image = self.image_trans(image)
                mask = self.transform(mask)
            
            self.allImgs[idx] = image.squeeze(0)
            self.allMasks[idx] = mask.squeeze(0)

        # self.final.to("cuda")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.allImgs[idx], self.allMasks[idx]
    

class BatchDataImages(Dataset):

    def __init__(self, imgPath, imgSize, imgC,imgTransform=None, num_images = None):

        self.transform = imgTransform
        self.imgs = os.listdir(imgPath)
        self.allImgs = torch.zeros([len(self.imgs),imgC, imgSize,imgSize], dtype=torch.float32)

        if num_images == None:
            num_images = len(self.imgs)
        for idx in tqdm(range(num_images)):
            
            img_path = os.path.join(imgPath, self.imgs[idx])

            # Opens image in uint format
            image = io.imread(img_path)

            # toPILImage converts the image to (0,1) range
            if self.transform:
                image = self.transform(image)
            self.allImgs[idx] = image.squeeze(0)

        # self.final.to("cuda")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.allImgs[idx]

def get_data_loader(args):

    if args.dataset == 'ImageMask':
        image_size = args.image_size
        train_dataset = BatchDataImageMask(imgPath=args.imagePath,imgSize = image_size, 
                                imgC = args.imgC,
                                imgTransform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((image_size,image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                ]),
                                maskTransform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((image_size,image_size)),
                                    transforms.ToTensor(),
                                ]), maskExt="png", num_images = args.num_images,)

    
    if args.dataset == 'GAN':
        image_size = args.image_size
        train_dataset = BatchDataImages(imgPath=args.imagePath,imgSize = image_size, 
                                imgC = args.imgC,
                                imgTransform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((image_size,image_size)),                 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                                    num_images = args.num_images,)

        # train_dataset = dset.ImageFolder(root=dataroot,
        #                    transform=transforms.Compose([
        #                        transforms.Resize(image_size),
        #                        transforms.CenterCrop(image_size),
        #                        transforms.ToTensor(),
        #                        transforms.Normalize((0.5), (0.5)),
        #                    ]))
    

    return train_dataset