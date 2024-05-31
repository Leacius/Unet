import torch
from torch import nn
from pathlib import Path
from torch.optim import Adam
from unet_light import UNet_light as UNet
from dataset import MyDataset as datasets


if __name__ == '__main__':
    root = "./output"

    dataset = datasets(root, size=(64, 64))
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=16, 
                                             num_workers=4,
                                             pin_memory=True,
                                             drop_last=True,
                                             shuffle=True)
    
    model = UNet(5)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    model.train()
    save_path = Path("./chkp")
    save_path.mkdir(exist_ok=True)

    for epoch in range(50):
        for i, (data, label) in enumerate(dataloader):
            data = data.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}")
        
        torch.save(model.state_dict(), f"chkp/model_{epoch}.pth")
