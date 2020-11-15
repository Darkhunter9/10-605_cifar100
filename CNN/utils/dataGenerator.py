import os

import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

def dataGenerator(train=True, batch_size=32):
    """
    Load CIFAR100 dataset and return Pytorch DataLoader.

    Input
    -------
    train: return training/validating data, bool;
    batch_size: batch size, int;

    Output
    -------
    dataloader: Pytorch DataLoader.
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    dataset = CIFAR100(root=os.path.join(dir_path, '../../', 'data'),
                        train=train, 
                        download=True,
                        transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader