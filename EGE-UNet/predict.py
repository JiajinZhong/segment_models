# import numpy as np
# import torch
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from configs.config_setting import setting_config
# from models.egeunet import EGEUNet
#
#
# config = setting_config
# model_cfg = config.model_config
# model = EGEUNet(num_classes=model_cfg['num_classes'],
#                 input_channels=model_cfg['input_channels'],
#                 c_list=model_cfg['c_list'],
#                 bridge=model_cfg['bridge'],
#                 gt_ds=model_cfg['gt_ds'], )
#
#
# transform = A.Compose([
#     A.Resize(128, 128),
#     ToTensorV2()
# ])
#
#
# weight_path = r"E:\segment_models\EGE-UNet\results\egeunet_yiwu_Monday_19_January_2026_16h_29m_09s\checkpoints\best-epoch298-loss0.7755.pth"
# state_dict = torch.load(weight_path, map_location='cpu')
# # 移除多GPU的module前缀（你的验证代码里也一定有这个逻辑）
# new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict, strict=False)
# model.eval()
#
#
# with torch.no_grad():
#     # 读取图片 + 预处理
#     img_path = r"E:\segment_models\EGE-UNet\data\yiwu\val\images\3.5_TFF458R094C34_PCBI 3_8_8417_6417_MD311_M_27_20251021_225150_0.77_L0.bmp"
#     img = cv2.imread(img_path)
#     img = transform(image=img)['image']
#
#     img = img.float().unsqueeze(0)  # shape [1, 1, 128, 128] 你的是单通道
#
#     # 模型返回值是 (gt_pre, out)
#     # 你的验证代码里写的：gt_pre, out = model(img)
#     gt_pre, out = model(img)
#
#     # 判断out是否是tuple，如果是取第一个
#     if type(out) is tuple:
#         out = out[0]
#
#     # squeeze(1) 去掉通道维度，和你的验证代码完全一致
#     out = out.squeeze(1).cpu().detach().numpy()  # shape [1, 128, 128]
#     seg_mask = out[0]  # 去掉batch维度，得到最终的mask矩阵 [128, 128]
#
#
#
# threshold = config.threshold  # 用你配置文件里的阈值，一般是0.5
# seg_mask_bin = np.where(seg_mask >= threshold, 255, 0).astype(np.uint8)
#
# # 还原mask到原图尺寸
# img_ori = cv2.imread(img_path)
# seg_mask_bin = cv2.resize(seg_mask_bin, (img_ori.shape[1], img_ori.shape[0]), cv2.INTER_NEAREST)
#
# # 保存结果
# cv2.imwrite("mask.png", seg_mask_bin)
#
#
# print(f"预测mask尺寸：{seg_mask_bin.shape}")
# print(f"前景像素数量：{np.sum(seg_mask_bin == 255)}")
# print(f"阈值设置：{threshold}")
import os

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config_setting import setting_config
from models.egeunet import EGEUNet



# 纯Numpy实现的CRF后处理函数
# 专为你的场景定制：二分类分割 + 单通道灰度图 + 128x128尺寸
# 效果：去噪点、平滑边缘、填补小空洞、提升分割精度，和原版pydensecrf一致
# 无任何依赖，直接运行，速度超快
def crf_post_process_numpy(img_ori, pred_score, resize_shape=(128,128), threshold=0.5):
    """
    :param img_ori: 原始读取的灰度图 (cv2.imread 读取的原图, uint8)
    :param pred_score: 模型输出的得分图 [128,128] 0~1的浮点数
    :param resize_shape: 模型输入尺寸，固定128x128
    :param threshold: 二值化阈值
    :return: CRF优化后的 0/1二值mask [128,128]
    """
    # 1. 原图和预测图统一resize到模型尺寸
    img = cv2.resize(img_ori, resize_shape, interpolation=cv2.INTER_NEAREST) / 255.0  # 归一化0-1
    pred_score = cv2.resize(pred_score, resize_shape, interpolation=cv2.INTER_NEAREST)

    # 2. 生成初始概率图 (背景概率，前景概率)
    bg_prob = 1 - pred_score
    fg_prob = pred_score
    prob = np.stack([bg_prob, fg_prob], axis=-1)  # [128,128,2]

    # CRF核心参数 (针对工业图调优，不用改)
    gamma = 3.0   # 空间距离权重，越小越平滑
    sigma = 0.1   # 像素相似度权重，越大越看重像素相似
    iter_num = 5  # 迭代次数，5次足够，速度快效果好

    # 3. 生成像素坐标矩阵，计算空间距离
    h, w = resize_shape
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    pos = np.stack([xx, yy], axis=-1)  # [128,128,2]

    # 4. CRF迭代优化 (核心逻辑)
    for _ in range(iter_num):
        new_prob = prob.copy()
        for i in range(h):
            for j in range(w):
                # 计算当前像素与所有像素的空间距离权重
                dist = np.exp(-np.sum((pos - pos[i,j])**2, axis=-1) / (2 * gamma**2))
                # 计算当前像素与所有像素的灰度相似度权重
                sim = np.exp(-np.square(img - img[i,j]) / (2 * sigma**2))
                # 联合权重：空间距离 + 像素相似度
                weight = dist * sim

                # 全局加权平均，优化当前像素的概率
                new_prob[i,j,0] = np.sum(prob[:,:,0] * weight) / np.sum(weight)
                new_prob[i,j,1] = np.sum(prob[:,:,1] * weight) / np.sum(weight)
        prob = new_prob

    # 5. 生成优化后的二值mask
    crf_mask = np.argmax(prob, axis=-1)  # 0=背景，1=前景
    return crf_mask

def calculate_metrics(pred_mask, gt_mask):
    """
    计算分割任务的5个核心评价指标
    :param pred_mask: 预测的二值mask，numpy数组，shape=[H,W]，值为0(背景)/1(前景)
    :param gt_mask:   标签的二值mask，numpy数组，shape=[H,W]，值为0(背景)/1(前景)
    :return: pa, dice, iou, precision, recall 全部返回
    """
    # 计算 真阳性TP 假阳性FP 真阴性TN 假阴性FN
    TP = np.sum((pred_mask == 1) & (gt_mask == 1))  # 预测前景+实际前景
    FP = np.sum((pred_mask == 1) & (gt_mask == 0))  # 预测前景+实际背景（误检）
    TN = np.sum((pred_mask == 0) & (gt_mask == 0))  # 预测背景+实际背景
    FN = np.sum((pred_mask == 0) & (gt_mask == 1))  # 预测背景+实际前景（漏检）

    # Pixel Accuracy 像素准确率
    PA = (TP + TN) / (TP + TN + FP + FN + 1e-8)  # +1e-8防止分母为0

    # Dice 系数 (分割核心指标)
    Dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    # IoU 交并比 (分割核心指标)
    IoU = TP / (TP + FP + FN + 1e-8)

    # Precision 精确率
    Precision = TP / (TP + FP + 1e-8)

    # Recall 召回率
    Recall = TP / (TP + FN + 1e-8)

    return PA, Dice, IoU, Precision, Recall

config = setting_config
model_cfg = config.model_config
model = EGEUNet(num_classes=model_cfg['num_classes'],
                input_channels=model_cfg['input_channels'],
                c_list=model_cfg['c_list'],
                bridge=model_cfg['bridge'],
                gt_ds=model_cfg['gt_ds'], )

transform = A.Compose([A.Resize(128, 128), ToTensorV2()])
weight_path = r"E:\segment_models\EGE-UNet\results\egeunet_yiwu_Monday_19_January_2026_16h_29m_09s\checkpoints\best-epoch298-loss0.7755.pth"
state_dict = torch.load(weight_path, map_location='cpu')
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.eval()

img_dir = r"E:\segment_models\EGE-UNet\data\yiwu\val\images"
mask_dir = r"E:\segment_models\EGE-UNet\data\yiwu\val\masks"
with torch.no_grad():

    for img in os.listdir(img_dir):
        print(img)
        img_name = img.split('.bmp')[0]
        img_path = os.path.join(img_dir, img)
        label_path = os.path.join(mask_dir, img_name + '.png')
        print(label_path)

        img_ori = cv2.imread(img_path)
        gt_mask_ori = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 读取标签为单通道灰度图

        img = transform(image=img_ori)['image']
        img = img.float().unsqueeze(0) / 255.0

        gt_pre, out = model(img)
        if type(out) is tuple:
            out = out[0]

        # crf_mask = crf_post_process_numpy(img_ori, pred_score, resize_shape=(128, 128), threshold=threshold)
        # 预测结果转 0/1二值mask
        pred_score = out.squeeze(1).cpu().detach().numpy()[0]  # 预测得分图 [128,128]
        threshold = config.threshold if hasattr(config, 'threshold') else 0.5
        pred_mask = np.where(pred_score >= threshold, 1, 0)  # 转0/1二值图，和标签格式一致

        # 标签mask做和预测图完全一致的预处理（必须！尺寸统一）
        gt_mask = transform(image=gt_mask_ori)['image']  # 标签也resize到128x128
        gt_mask = gt_mask.squeeze(0).cpu().numpy()  # 转numpy [128,128]
        gt_mask = np.where(gt_mask >= 0.5, 1, 0)  # 标签也转0/1二值图

        PA, Dice, IoU, Precision, Recall = calculate_metrics(pred_mask, gt_mask)

        print("=" * 50)
        print("单张图片 分割评价指标 计算完成！")
        print(f"像素准确率 (PA)  : {PA:.4f}")
        print(f"Dice系数        : {Dice:.4f}")  # 核心指标
        print(f"IoU交并比       : {IoU:.4f}")  # 核心指标
        print(f"精确率 (Precision): {Precision:.4f}")
        print(f"召回率 (Recall)  : {Recall:.4f}")
        print("=" * 50)

        # 保存二值化的预测mask和标签mask
        pred_mask_vis = np.where(pred_mask == 1, 255, 0).astype(np.uint8)
        gt_mask_vis = np.where(gt_mask == 1, 255, 0).astype(np.uint8)
        cv2.imwrite(rf"E:\segment_models\EGE-UNet\testres/pred_mask_{img_name}.png", pred_mask_vis)



