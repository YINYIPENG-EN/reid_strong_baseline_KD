from modeling import Baseline
import torch
model = Baseline(751, 1, '../weights/resnet50-19c8e357.pth', 'resnet50_kd', 'bnneck', 'after', True)
x = torch.zeros(8, 3, 256, 128)
_, _, out = model(x)
print(out[0].shape)
