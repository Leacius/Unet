import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt

def transform(image, mask):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std), 
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),
                ])
    
    transform_mask = transforms.Compose([
                transforms.Resize((256, 256)),
                # transforms.RandomHorizontalFlip(),
                ])
    
    image = transform(image)
    mask = torch.from_numpy(mask).float()
    mask = transform_mask(mask)

    image = np.array(image)
    mask = np.array(mask)

    return image, mask

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes=5):
        self.root = Path(root)
        self.data = list((self.root / "JPEGImages").glob('*.jpg'))
        self.label = list((self.root / "SegmentationClass").glob('*.npy'))
        self.classes = classes

    def load_image(self, path):
        path = str(path)
        imgs = Image.open(path).convert('RGB')

        return imgs
    
    def load_label(self, path):
        path = str(path)
        label = np.load(path)
        maps = np.zeros((self.classes, label.shape[0], label.shape[1]))
        for i in range(1, 5):
            if (label == i).any():
                maps[i][label == i] = 1

        return maps

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.load_image(self.data[idx])
        label = self.load_label(self.label[idx])

        data, label = transform(data, label)
        
        return data, label
    
if __name__ == '__main__':
    root = "./output"
    dataset = MyDataset(root)
    data, label = dataset[0]

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(data[0])
    fig.add_subplot(1, 2, 2)
    plt.imshow(label[2])
    plt.show()
    plt.close()