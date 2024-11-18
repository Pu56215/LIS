# from torchsummary import summary
# from mmdet.models.backbones.resnet import ResNetAdaDSmoothPrior

# # 创建模型
# model = ResNetAdaDSmoothPrior(style='caffe',depth=50).cuda()

# # 确保输入大小匹配
# input_size = (3, 1280, 960)  # 例如你需要使用的输入尺寸

# # 测试模型结构
# summary(model, input_size=input_size)


import torch
from mmdet.models.backbones.transnext_native import transnext_small

# 创建模型
model = transnext_small(pretrained=None, img_size=640, pretrain_size=224,embed_dims=[256,512,1024,2048]).cuda()

# 确保输入大小匹配
input_size = (3, 640, 480)  # 例如你需要使用的输入尺寸

# 创建与模型输入大小匹配的随机张量
x = torch.randn(2, *input_size).cuda()  # Batch size 为 2

# 运行模型前向传播
output = model(x)

# 打印输出的形状，确认模型输出
print(f"Output shape: {output[-1].shape}")
