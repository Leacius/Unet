import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from torchvision import transforms
from unet_light import UNet_light as UNet


def draw_mask(img, mask, name="mask.jpg"):
    colors = [(0, 0, 0), (255, 0, 0), (0, 200, 55), (50, 127, 127), (100, 100, 255)]
    maps = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        idx = mask == i
        maps[:,:,0][idx] = colors[i][0]
        maps[:,:,1][idx] = colors[i][1]
        maps[:,:,2][idx] = colors[i][2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(maps)
    plt.savefig(name)
    plt.show()

if __name__ == '__main__':

    size = (64, 64)

    model = UNet(5)
    model_state = torch.load("./chkp/model_31.pth")
    model.load_state_dict(model_state)
    model.eval()
    model.cuda()

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize(size),
                ])
    
    img = Image.open("0036.jpg").convert('RGB')
    img = transform(img)
    ori_img = img.clone().numpy().transpose(1, 2, 0)
    img = img.unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(img)
        out = out.argmax(axis=1)
        out = out.squeeze(0).cpu().numpy()

        draw_mask(ori_img, out, "epoch_21_test.jpg")

        