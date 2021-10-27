from train import train
from test import test
import torch.nn as nn
import torch
from data import Data
from feedforward import ResNet18
import torch.optim as optim
trainloader,testloader=Data().CIFAR10()
from torch.optim.lr_scheduler import ReduceLROnPlateau


from tqdm import tqdm

def run(epochs,file_name,lr):
    tr,te=[],[]
    criterion = nn.CrossEntropyLoss()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    best_accuracy = 0
    for epoch in range(1,epochs):
      train_average_loss,train_accuracy = train(epoch,net,device,trainloader,optimizer,criterion)
      test_average_loss,test_accuracy = test(epoch,net,testloader,device,criterion)
      te.append(test_accuracy)
      tr.append(train_accuracy)
      print(f'\nEpoch: {epoch}\n\t Train set: Average loss: {train_average_loss}, Accuracy: {train_accuracy}%')
      print(f'\n\tTest set: Average loss: {test_average_loss}, Accuracy: {test_accuracy}%')
      if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        checkpoint = {'state-dict':net.state_dict(),'optimizer':optimizer.state_dict()}
        torch.save(checkpoint,file_name)
      scheduler.step(test_average_loss)
    return tr,te