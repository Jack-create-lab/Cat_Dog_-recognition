import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 定义数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 定义类别名称（根据训练时的类别）
class_names = ['cat', 'dog']  # 这个应该与训练时的类别一致

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft.load_state_dict(torch.load('model_cat_dog.pth'))
model_ft = model_ft.to(device)
model_ft.eval()  # 设置为评估模式

def predict_image(image_path, model, data_transforms, class_names, device):
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]

# 预测
image_path = '1.jpg'
prediction = predict_image(image_path, model_ft, data_transforms, class_names, device)
print(f'Predicted: {prediction}')
