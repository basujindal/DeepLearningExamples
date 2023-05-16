
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as data_utils
import wandb
import custom_loaders
import conv_layers


latent_len = 100
img_size = 64
n_channels = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
channelsG = [latent_len, 256, 128, 128,64, 32, n_channels]
channelsD = [n_channels, 32, 64, 128, 128,256, 1]
label_flip = 0
add_noise = 0
bs = 32
log_iter = 200
log = True

G_lr = 0.0002
D_lr = 0.0002
epochs = 20
D_epochs = 1


assert(len(channelsD) == len(channelsG))
assert(img_size == 2**(len(channelsD) - 1))


class Args():
    def __init__(self):
        self.dataset = 'GAN'
        # self.imagePath = '/root/data/data/JSRT/Images'
        self.imagePath = '/root/celeba/img_align_celeba/'
        self.image_size = img_size
        self.download = False
        self.imgC = n_channels
        self.num_images = None

args = Args()
print("Loading data...")
train_dataset = custom_loaders.get_data_loader(args)
train_loader = data_utils.DataLoader(train_dataset, batch_size=bs, shuffle=True)

G = conv_layers.GeneratorTrans(channelsG).to(device)
D = conv_layers.Discriminator(channelsD).to(device)
# G = conv_layers.GeneratorUpSample(channelsG).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(G.parameters(), lr=G_lr, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(D.parameters(), lr=D_lr, betas=(0.5, 0.999))
fixed_noise = torch.rand(bs,latent_len,1,1).to(device)


config={"epochs": epochs, "batch_size": bs,
         "D_epochs": D_epochs, "D_lr": D_lr, "G_lr": G_lr,
           "img_size": img_size, "n_channels": n_channels,
           "latent_len": latent_len}

wandb.init(project='pytorch-gan-celeba', entity='basujindal123', config=config)

lossD_Real = 0
lossD_Fake = 0
lossG = 0
iter = 0


for i in (range(epochs)):
    for data in tqdm(train_loader):        
        real_imgs = data.to(device)

        iter+=1
        # Training Discriminator
        D.zero_grad()

        with torch.no_grad():
            z = torch.rand(bs,latent_len,1,1).to(device)
            fake_imgs = G(z)

        output = D(fake_imgs).flatten()

        label_val = 0
        ## randomly flip labels
        if label_flip and np.random.random() > 0.95:
            label_val = 1

        fake_labels = np.array([label_val]*output.shape[0])
        
        if add_noise:
            fake_labels = fake_labels + np.random.normal(0,0.05,fake_labels.shape[0])
        labels = torch.tensor(fake_labels).float().to(device)
        lossF = criterion(output, labels)
        lossF.backward()

        real_imgs = real_imgs.to(device)
        output = D(real_imgs).flatten()

        label_val = 1
        ## randomly flip labels
        if label_flip and np.random.random() > 0.95:
            label_val = 0
            
        real_labels = np.array([label_val]*output.shape[0])
        
        if add_noise:
            real_labels = real_labels + np.random.normal(0,0.05,real_labels.shape[0])
        labels = torch.tensor(real_labels).float().to(device)
        lossR = criterion(output, labels)
        lossR.backward()


        lossD = lossR + lossF
        optimizerD.step()

        lossD_Real+=lossR.item()
        lossD_Fake+=lossF.item()

        # if((iter+1)%D_epochs == 0):
        if 1:
            ## Training Generator
            G.train()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            z = torch.rand(bs,latent_len,1,1).to(device)
            fake_imgs = G(z)
            output = D(fake_imgs).flatten()

            label = torch.tensor([1]*output.shape[0]).float().to(device)
            lossG = criterion(output, label)
            lossG.backward()
            optimizerG.step()
            lossG = lossG.item()


        if((iter+1)%log_iter == 0 and log==True):

            G.eval()
            with torch.no_grad():
                fixed_fake_imgs = G(fixed_noise[:16]).detach()

            wandb.log({
                'lossG': lossG,
                'lossD_Real': lossD_Real,
                'lossD_Fake': lossD_Fake,
                'lossD': lossD_Real + lossD_Fake,
                'Fake Images': [wandb.Image(i) for i in fixed_fake_imgs],
                'Real Images' : [wandb.Image(i) for i in real_imgs[:16].detach()]
                })
            
            lossD_Real = 0
            lossD_Fake = 0


