import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torch.nn as nn
from PIL import Image
import os
import torchvision.transforms as T
from typing import Any, Tuple

# 自定义数据集路径和JSON注释文件
data_dir = "dataset"
train_json = "train_with_coco_json/annotations.json"
val_json = "val_with_coco_json/annotations.json"

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack([img for img in images], dim=0)
    targets = [{k: torch.tensor(v) for k, v in t.items()} for t in targets]
    return images, targets


# 定义自定义转换
class ToTensorWithTarget(object):
    def __call__(self, image, target):
        return torchvision.transforms.functional.to_tensor(image), target
class CustomCocoDetection(CocoDetection):
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = path.replace("JPEGImages/", "")  # 删除额外的 "JPEGImages/" 部分
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target_data = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[id]))

        boxes = []
        labels = []
        for ann in target_data:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class ComposeTransforms:
    def __init__(self, image_transforms, target_transforms=None):
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def __call__(self, image, target):
        image = self.image_transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return image, target



# 数据预处理

transforms = ComposeTransforms(
    image_transforms=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    target_transforms=None  # 如果需要，您也可以在这里添加针对目标的转换
)

# 加载数据集
train_dataset = CustomCocoDetection(data_dir + "/train_with_coco_json/JPEGImages", data_dir + "/" + train_json, transforms=transforms)
val_dataset = CustomCocoDetection(data_dir + "/val_with_coco_json/JPEGImages", data_dir + "/" + val_json, transforms=transforms)

batch_size = 4  # 您可以根据需要调整批量大小
num_workers = 4  # 您可以根据需要调整工作线程数

# 创建数据加载器
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)


class SimpleConvNet(nn.Module):
    def __init__(self, out_channels):
        super(SimpleConvNet, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x


# 设置输出通道数为256
out_channels = 256
backbone = SimpleConvNet(out_channels)

# 创建Faster R-CNN模型
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

model = FasterRCNN(backbone,
                   num_classes=len(val_dataset.coco.getCatIds()) + 1,  # 类别数量加1，因为0是背景
                   rpn_anchor_generator=anchor_generator)



# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 设置优化器和学习率调度器
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
if __name__ == '__main__':
    # 训练参数
    num_epochs = 10

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in train_dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict[k] for k in loss_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        # 更新学习率
        lr_scheduler.step()

        # 在验证集上评估模型性能
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # 计算验证集损失
                loss_dict = model(images, targets)
                losses = sum(loss_dict[k] for k in loss_dict)

                val_total_loss += losses.item()

            # 添加评价指标，如mAP等
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_dataloader)}, Val Loss: {val_total_loss / len(val_dataloader)}")

    torch.save(model.state_dict(), "faster_rcnn_model.pth")
