from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv.runner import load_checkpoint
import os

def main():
    # 读取配置文件
    cfg = Config.fromfile('Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS.py')

    # 设置GPU和batch size
    cfg.gpu_ids = range(1)  # 1 表示使用单个 GPU
    BATCHSIZE = 8
    cfg.data.samples_per_gpu = BATCHSIZE

    # 数据集的根目录
    cfg.data_root = '/home/WorkSpace_newnew/pbx/LIS/dataset/coco/'
    
    # 加载预训练模型
    cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth'

    # 设置工作目录
    cfg.work_dir = './work_dirs/mask_rcnn_coco'

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 构建模型
    model = build_detector(cfg.model)
    model.init_weights()
    
    # 开始训练
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()
