from torchvision import datasets, transforms
import torch

class Data:
    def __init__(self):
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

    def MNIST(self):
        train = datasets.MNIST(root="./data",train=True,transform= self.transformation, download=True)
        test = datasets.MNIST(root="./data",train=False,transform= self.transformation, download=True)
        seed = 1
        cuda = torch.cuda.is_available()
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(
            shuffle=True, batch_size=64)
        train_loader = torch.utils.data.DataLoader(dataset=train, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(dataset=test, **dataloader_args)
        return train_loader, test_loader