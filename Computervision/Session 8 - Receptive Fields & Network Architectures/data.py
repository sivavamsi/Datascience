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
    def CIFAR10(self):
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        trainset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        
        testset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader,testloader