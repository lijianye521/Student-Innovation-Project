import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

# 自定义数据集路径和JSON标注文件
data_dir = "path/to/your/dataset"
train_json = "train_with_coco_json/annotations.json"
val_json = "val_with_coco_json/annotations.json"

# 数据预处理
transforms = ToTensor()

# 加载数据集
train_dataset = CocoDetection(data_dir + "/train_with_coco_json/JPEGImages", data_dir + "/" + train_json, transforms=transforms)
val_dataset = CocoDetection(data_dir + "/val_with_coco_json/JPEGImages", data_dir + "/" + val_json, transforms=transforms)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

# 加载预训练的ResNet50模型
backbone = torchvision.models.resnet50(pretrained=True)
backbone_out_features = backbone.fc.in_features

# 删除全连接层以获得特征提取器
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

# 创建Faster R-CNN模型
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
model = FasterRCNN(backbone,
                   num_classes=len(train_dataset.coco.getCatIds()) + 1,  # 类别数量加1，因为0是背景
                   rpn_anchor_generator=anchor_generator)

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 设置优化器和学习率调度器
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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
        losses = sum(loss for loss in loss_dict.values())

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
            losses = sum(loss for loss in loss_dict.values())

            val_total_loss += losses.item()

            # 这里可以
        # 添加评价指标，如mAP等
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_dataloader)}, Val Loss: {val_total_loss / len(val_dataloader)}")

torch.save(model.state_dict(), "faster_rcnn_model.pth")
