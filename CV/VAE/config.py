img_size = 256
n_channels = 1
epochs = 10
iter = 0
log = False
log_iter = 10
bs = 4
device = "cuda"
latent_len = 256
kl_weight = 0.3

## Data Loader Config

class Args:
    def __init__(self):
        self.dataset = 'GAN'
        self.imagePath = '/home/thunder/OneDrive/quarter3/ECE285_DeepGen/HW3/chest_xray/train/PNEUMONIA'
        self.image_size = img_size
        self.download = False
        self.num_images = None
        self.imgC = n_channels
        self.convert2bw = True

train_args = Args()

test_args = Args()
test_args.imagePath = '/home/thunder/OneDrive/quarter3/ECE285_DeepGen/HW3/chest_xray/test/PNEUMONIA'

