import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms

# 全局变量
num_class = 2

# 数据集
valid_trainsform = transforms.Compose([
    transforms.Resize(214),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

valid_datasets = ImageFolder('./data/test/', transform=valid_trainsform)
valid_loader = DataLoader(valid_datasets, batch_size=1, shuffle=True)
print(valid_datasets.class_to_idx)

# 模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, num_class)
model.load_state_dict(torch.load('./model@acc1.000.pth'))
model.eval()
model.cuda()


for inputs, targets in valid_loader:
    # 前向传播
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = model(inputs)
    # 统计正确个数
    _, preds = torch.max(outputs.data, 1)
    print('target: ', targets.item(), end=' | ')
    print('pred :', preds.item())

