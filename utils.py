from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


def Load_image(imagepath, imsize=512):    
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()]) 
    image = Image.open(imagepath)
    image = loader(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return image.to(device, torch.float)

def Show_images(tensors):
    unloader = transforms.ToPILImage()
    if len(tensors)==2:
        fig = plt.figure(figsize=(10,20))
    else:
        fig = plt.figure(figsize=(10,30))
    for i,tens in enumerate(tensors):
        #plt.figure(figsize=(10,30))
        plt.subplot(1, len(tensors), i+1)
        plt.axis("off")
        plt.title(tens[1])
        image = tens[0].cpu().clone()   
        image = image.squeeze(0)
        image = unloader(image)
        plt.imshow(image)
    plt.show();

def Save_image(tensor,filename):
    out = tensor.squeeze(0)
    out = out.cpu().detach().numpy()
    out = np.moveaxis(out,0,2)
    plt.imsave(filename,out)
    
        
def Make_gif(imgs,filename):
    unloader = transforms.ToPILImage()
    gif_list=[]
    fig = plt.figure(figsize=(7,7),frameon=False)
    plt.axis("off")
    for im in imgs:
        image = im.cpu().detach().clone()
        image = image.squeeze(0)
        image = unloader(image)
        gif_list.append([plt.imshow(image,animated=True)])
    animation = anim.ArtistAnimation(fig,gif_list)
    animation.save(filename,writer='imagemagick')
    
    
    
    
    
    
    
    
    
    
    
    
    
    