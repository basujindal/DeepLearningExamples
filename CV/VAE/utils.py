import torch
import matplotlib.pyplot as plt
from ignite.metrics import FID, InceptionScore
from ignite.engine import Engine
import PIL.Image as Image
from torchvision import transforms

    
def get_reconstructed(model, loader, device, num_imgs=4):
    num_imgs = 5

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            x_hat, _, _ = model(data)
            break


    return data[:num_imgs], x_hat[:num_imgs]


def show_images(model, device, test_loader, num_imgs=8):

    sample = model.sample_images(num_imgs)
    data, recon = get_reconstructed(model, test_loader, device)

    fig, axs = plt.subplots(2, 8, figsize=(30, 7))

    for j in range(num_imgs):
        axs[0, j].imshow(sample[j].numpy().squeeze(), cmap='gray')

        if j%2 == 0:
            axs[1, j].imshow(data[j//2].cpu().numpy().squeeze(), cmap='gray')
        else:
            axs[1, j].imshow(recon[j//2].cpu().numpy().squeeze(), cmap='gray')

    
    ## mention on plot 1st 2 columns are samples, 3rd column is data, 4th column is recon

    rows = ['Sampled image', 'Training and Reconstructed']

    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row)

    plt.show()
    

def PILinterpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        img = transforms.ToTensor()(resized_img)
        img = img.repeat(3, 1, 1)
        arr.append(img)
        
    ans = torch.stack(arr)
    return ans
    

def get_fid(loader, device, evaluation_step): 

    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])
    evaluator = Engine(evaluation_step)


    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    print("Starting evaluation...")

    evaluator.run(loader, max_epochs=1) # use your test data loader, NOT training data loader
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']

    print("FID:", fid_score)
    print("Inception score:", is_score)

    return fid_score, is_score

def plot_losses_and_fid(losses, fid_scores):

    fig, axs = plt.subplots(1,4, figsize=(30, 7))
    axs[0].plot([x[0] for x in losses])
    axs[0].set_title('KL loss')
    axs[1].plot([x[1] for x in losses])
    axs[1].set_title('MSE loss')
    axs[2].plot([x[0] for x in fid_scores])
    axs[2].set_title('FID')
    axs[3].plot([x[1] for x in fid_scores])
    axs[3].set_title('IS')