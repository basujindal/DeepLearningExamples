import torch
import torch.nn as nn


class UpTranspose2d(nn.Module):

    '''
    Doubles input image size
    Stride assumed to be 2 and padding depends on kernel size.
    Leaky ReLu generally used in GANs & BatchNorm does not preserve the independence between images,
    therefore instance norm used.
    '''

    def __init__(self, ch_in, ch_out, kernelSize = 4, leak = 0.2, stride = 2):
        super().__init__()

        self.upTrans = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernelSize, stride = stride, padding=(kernelSize//2 - 1))
        # self.norm = nn.InstanceNorm2d(ch_out, affine=True)
        # self.actFn = nn.LeakyReLU(leak, inplace=True)
        self.norm = nn.BatchNorm2d(ch_out)
        self.actFn = nn.ReLU()
        
    def forward(self, image):

        x = self.upTrans(image)
        x = self.actFn(self.norm(x))
        return x

class UpSampleConv(nn.Module):

    '''
    Doubles input image size
    Nearest Neighbour Upsample doubles image size followed by two Conv Layers as used in ResNet
    '''

    def __init__(self, ch_in, ch_out,kernelSize = 3, mode = 'nearest'):
        super().__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=kernelSize, padding = (kernelSize-1)//2)
        self.norm1 = nn.BatchNorm2d(ch_out, affine=True)
        self.actFn1 = nn.ReLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=kernelSize, padding = (kernelSize-1)//2)
        self.norm2 = nn.BatchNorm2d(ch_out, affine=True)
        self.actFn2 = nn.ReLU()
        
    def forward(self, image):

        x = self.actFn1(self.norm1(self.conv1(image)))
        x = self.actFn2(self.norm2(self.conv2(x)))
        return x
    
class GeneratorTrans(nn.Module):

    def __init__(self, channels, kernelSize=4):
        super().__init__()

        self.gen = nn.ModuleList([UpTranspose2d(channels[i], channels[i+1], kernelSize) for i in range(len(channels) - 2)])
        self.output = nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size=kernelSize, stride = 2, padding=(kernelSize//2 - 1))

    def forward(self, image):
            
        for block in self.gen:
            image = block(image)
    
        output = torch.tanh(self.output(image))
        return output

class GeneratorUpSample(nn.Module):

    def __init__(self, channels, kernelSize=3):
        super().__init__()

        self.gen = nn.ModuleList([UpSampleConv(channels[i], channels[i+1], kernelSize) for i in range(len(channels) - 2)])
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest') 
        self.output = nn.Conv2d(channels[-2], channels[-1], kernel_size=kernelSize, padding = (kernelSize-1)//2)


    def forward(self, image):

        for block in self.gen:
            image = block(self.upSample(image))

        output = torch.tanh(self.output(self.upSample(image)))
        return output

class DownConv2d(nn.Module):

    '''
    Halves input image size
    Using stride = 2 instead of Pooling.
    '''

    def __init__(self, ch_in, ch_out,kernelSize):
        super().__init__()

        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernelSize, stride = 2, padding = kernelSize//2 - 1)
        # self.norm = nn.InstanceNorm2d(ch_out, affine=True)
        # self.actFn = nn.ReLU()
        self.norm = nn.BatchNorm2d(ch_out)
        self.actFn = nn.LeakyReLU(0.2)
        
    def forward(self, image):

        x = self.actFn(self.norm(self.conv(image)))
        return x

class Discriminator(nn.Module):
    def __init__(self, channels, kernelSize=4):
        super().__init__()

        self.dis = nn.ModuleList([DownConv2d(channels[i], channels[i+1], kernelSize) for i in range(len(channels) - 2)])
        self.out = nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=kernelSize, stride = 2, padding = kernelSize//2 - 1)
        self.sig = nn.Sigmoid()
    def forward(self, image):
        
        for block in self.dis:
            image = block(image)
        
        output = self.sig(self.out(image))
        return output