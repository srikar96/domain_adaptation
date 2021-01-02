from torch.autograd import Function
from torch import nn
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
import numpy as np
# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None

class DACNN(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.feature_extractor = nn.Sequential(
             # 1st conv layer
             # input [1 x 28 x 28] new [3 x 224 x 224]
             # output [20 x 12 x 12] new [32 x 112 x 112]
              nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride = 2),
              nn.BatchNorm2d(32),
              # 2nd conv layer
              # input [20 x 12 x 12] new [32 x 112 x 112]
              # output [50 x 4 x 4] new [64 x 56 x 56]
              nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride = 2),
              nn.BatchNorm2d(64),
              # 3rd conv layer
              # input  [64 x 56 x 56]
              # output [64 x 28 x 28]
              nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride = 2),
              nn.BatchNorm2d(64),
              # 4th conv layer
              # input  [64 x 28 x 28]
              # output [64 x 14 x 14]
              nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride = 2),
              nn.BatchNorm2d(64),
              nn.Dropout2d(0.2))
        '''

        #self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        #self.feature_extractor = torchvision.models.googlenet(pretrained=True)

        self.feature_extractor = torchvision.models.resnet50(pretrained=True)

        ct = 0
        for child in self.feature_extractor.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

        modules = list(self.feature_extractor.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        self.class_classifier = nn.Sequential(
            nn.Linear(2048, 10),
            #nn.ReLU(True),
            #nn.Dropout(0.2),
            #nn.Linear(512, 128), nn.BatchNorm1d(128),
            #nn.ReLU(True),
            #nn.Dropout(0.2),
            #nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, grl_lambda=1.0):
        # Handle single-channel input by expanding (repeating) the singleton dimention
        # x = x.expand(x.data.shape[0], 3, image_size, image_size)

        features = self.feature_extractor(x)
        #print(features.shape)
        features = features.view(-1, 2048)
        reverse_features = GradientReversalFn.apply(features, grl_lambda)

        class_pred = self.class_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)
        return class_pred, domain_pred

lr = 1e-3
n_epochs = 200

# Setup optimizer as usual
model = DACNN().cuda()
optimizer = optim.Adam(model.parameters(), lr)

# Two losses functions this time
loss_fn_class = torch.nn.NLLLoss()
loss_fn_domain = torch.nn.NLLLoss()

data_path = "./datasets/domain_lim/train/"
data_path_val = "./datasets/domain_lim/validation/"

batch_size = 64

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

dl_source = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)
dl_target = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle=True)

def train():
    # We'll train the same number of batches from both datasets
    max_batches = min(len(dl_source), len(dl_target))

    #model = DACNN().cuda()
    #model.load_state_dict(torch.load("./../checkpoints/domain_adapt.pt"))

    for epoch_idx in range(n_epochs):
        print(f"Epoch {epoch_idx+1:04d} / {n_epochs:04d}")
        dl_source_iter = iter(dl_source)
        dl_target_iter = iter(dl_target)

        for batch_idx in range(max_batches):
            optimizer.zero_grad()
            # Training progress and GRL lambda
            p = float(batch_idx + epoch_idx * max_batches) / (n_epochs * max_batches)
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1

            # Train on source domain
            X_s, y_s = next(dl_source_iter)
            # print(X_s.shape)
            # break
            y_s_domain = torch.zeros(X_s.shape[0], dtype=torch.long) # generate source domain labels

            class_pred, domain_pred = model(X_s.cuda(), grl_lambda)
            loss_s_label = loss_fn_class(class_pred, y_s.cuda())
            loss_s_domain = loss_fn_domain(domain_pred.cuda(), y_s_domain.cuda())

            # Train on target domain
            X_t, _ = next(dl_target_iter) # ignore target domain class labels!
            y_t_domain = torch.ones(X_t.shape[0], dtype=torch.long) # generate target domain labels

            _, domain_pred = model(X_t.cuda(), grl_lambda)
            loss_t_domain = loss_fn_domain(domain_pred, y_t_domain.cuda())

            loss = loss_t_domain + loss_s_domain + loss_s_label
            loss.backward()
            optimizer.step()

            print(f'[{batch_idx+1}/{max_batches}] '
                  f'class_loss: {loss_s_label.item():.4f} ' f's_domain_loss: {loss_s_domain.item():.4f} '
                  f't_domain_loss: {loss_t_domain.item():.4f} ' f'grl_lambda: {grl_lambda:.3f} '
                 )
            # if batch_idx == 2:
            #     print('This is just a demo, stopping...')
            #     break
        torch.save(model.state_dict(), "./../checkpoints/domain_adapt.pt")
        torch.save(optimizer.state_dict(), "./../checkpoints/domain_adapt_optimizer.pt")

def test():
    model = DACNN().cuda()
    model.load_state_dict(torch.load("./../checkpoints/domain_adapt.pt"))
    model.eval()

    print('Testing...')
    grl_lambda = 1
    """
    pred_test = 0
    for batch in dl_source_val:
        inputs, target = batch
        inputs = inputs.cuda()
        target = target.cuda()
        class_pred, _ = model(inputs, grl_lambda)
        pred_test += (torch.argmax(class_pred, dim = 1) == target).float().sum()
    correct = pred_test
    test_accuracy = (correct/len(dl_source_val.dataset))*100
    print('Source Accuracy: {:.2f}%\n'.format(test_accuracy))
    """
    pred_test = 0
    for batch in dl_target:
        inputs, target = batch
        inputs = inputs.cuda()
        target = target.cuda()
        class_pred, _ = model(inputs, grl_lambda)
        pred_test += (torch.argmax(class_pred, dim = 1) == target).float().sum()
    correct = pred_test
    test_accuracy = (correct/len(dl_target.dataset))*100
    print('Target Accuracy: {:.2f}%\n'.format(test_accuracy))

    return None

#train()
test()

#model = DACNN()
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))
