import gzip
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch import nn
from collections import Counter

assert torch.cuda.is_available()
data_path = "./datasets/domain_lim/train/"
data_path_val = "./datasets/domain_lim/validation/"

batch_size = 512

pre_process_train = transforms.Compose([
    transforms.Resize(300), # TODO: Resize
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

pre_process_val = transforms.Compose([
    transforms.Resize(300), # TODO: Resize
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=data_path,transform=pre_process_train)
val_data = torchvision.datasets.ImageFolder(root=data_path_val,transform=pre_process_val)

syn_data_loader_train = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)
syn_data_loader_val = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle=True)

class My_simpNet(nn.Module):
    def __init__(self):
        super(My_simpNet, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)

        for p in self.resnet50.parameters():
            p.requires_grad = False

        # for p in resnet50.layer4.fc.parameters():
        #     p.requires_grad = True

        n_classes = 10
        self.resnet50.fc = nn.Sequential(
            #nn.Linear(2048, 1024),
            #nn.ReLU(True),
            #nn.Linear(1024, 512),
            #nn.ReLU(True),
            nn.Linear(2048, n_classes))


    def forward(self, x):
        output = self.resnet50(x)
        return output

model = My_simpNet().cuda()
# print(model)
# params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
src_opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()

#model.load_state_dict(torch.load("./My_simpNet1.pt"))
#src_opt.load_state_dict(torch.load("./My_optimizer1.pt"))

#print(model, src_opt)

def validate_dataset():
    model.load_state_dict(torch.load("./../checkpoints/My_simpNet1.pt"))
    src_opt.load_state_dict(torch.load("./../checkpoints/My_optimizer1.pt"))
    model.eval()
    pred_test = 0
    print('In Val')
    for batch in syn_data_loader_val:
        inputs, target = batch
        inputs = inputs.cuda()
        target = target.cuda()
        prediction = model(inputs)
        pred_test += (torch.argmax(prediction, dim = 1) == target).float().sum()
    correct = pred_test
    test_accuracy = (correct/len(syn_data_loader_val.dataset))*100
    print('Accuracy: {:.2f}%\n'.format(test_accuracy))

def train():
    for e in range(100):
        loss_sum = 0.0
        pred = 0
        for step, (images, labels) in enumerate(syn_data_loader_train):
            images = images.cuda()
            labels = labels.squeeze_().cuda()

            src_opt.zero_grad()

            predicted = model(images)

            loss = loss_func(predicted, labels)
            pred += (torch.argmax(predicted, dim = 1) == labels).float().sum()

            loss_sum += loss.item()

            loss.backward()
            src_opt.step()

            #print('Done with step: {}'.format(step))
        loss = loss_sum / step
        print("Source ({}) - Epoch [{}/{}] - loss={:.2f} - Accuracy={}".format(type(syn_data_loader_train).__name__, e+1, 100, loss, pred/len(syn_data_loader_train.dataset)))

        torch.save(model.state_dict(), "./../checkpoints/My_simpNet1.pt")
        torch.save(src_opt.state_dict(), './../checkpoints/My_optimizer1.pt')
    return None

validate_dataset()
#train()
