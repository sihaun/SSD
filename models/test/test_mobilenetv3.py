import sys
import os
import torch
sys.path.append(os.path.abspath("./"))

from models.mobilenet import create_mobilenetv3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

NUM_CLASSES = 1
model = create_mobilenetv3(NUM_CLASSES).to(device)
x=torch.randn(1,3,224,224).to(device)
y=model(x)
print(y)