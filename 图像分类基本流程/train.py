import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms

# 全局变量
num_class = 2
batch_size = 5
epoch = 200

# 数据集
train_transform = transforms.Compose([
    transforms.Resize(214),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

valid_trainsform = transforms.Compose([
    transforms.Resize(214),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

train_datasets = ImageFolder('./dataset/train', transform=train_transform)
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
valid_datasets = ImageFolder('./dataset/valid', transform=valid_trainsform)
valid_loader = DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

# fine ture模型
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, num_class)
for m in model.modules():
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0, 0.01)
        m.bias.data.zero_()
model.cuda()

# 优化策略
fc_parameters = list(map(id, model.fc.parameters()))
conv_parameters = (p for p in model.parameters() if id(p) not in fc_parameters)
parameters = [{'params': conv_parameters, 'lr': 1e-4},
              {'params': model.fc.parameters()}]
optimizer = torch.optim.SGD(parameters, lr=1e-3, momentum=0.9, weight_decay=1e-4)

# 损失函数
criterion = nn.CrossEntropyLoss()


def train():
    model.train()
    num_correct = 0
    total_loss = 0
    for inputs, targets in train_loader:
        # 前向传播
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        # 统计正确个数
        _, preds = torch.max(outputs.data, 1)
        num_correct += torch.sum(preds == targets.data).item()
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_epoch = total_loss / len(train_loader)
    acc_epoch = num_correct / len(train_datasets)
    print('loss : %.4f | train_acc : %.4f' % (loss_epoch, acc_epoch), end=' | ')


best_valid_acc = 0


def valid():
    global best_valid_acc
    model.eval()
    num_correct = 0
    for inputs, targets in train_loader:
        # 前向传播
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        # 统计正确个数
        _, preds = torch.max(outputs.data, 1)
        num_correct += torch.sum(preds == targets.data).item()
    acc_epoch = num_correct / len(train_datasets)
    print('test_acc : %.4f' % acc_epoch)
    # 保存模型
    if acc_epoch > 0.9 and acc_epoch > best_valid_acc:
        torch.save(model.state_dict(), 'model@acc%.3f.pth' % acc_epoch)
        best_valid_acc = acc_epoch


for i in range(epoch):
    print('Epoch : {}'.format(i), end=' | ')
    train()
    valid()
