import os
import json
import numpy as np
import cv2


JSON_DIR = r"E:\A_TCL_data\M3B\M3B_PCBI\segment_datas\selected_data\selected_labels"  # 你的json标签文件夹路径
LABEL_MAP = {"yiwu": 255}  # 标签名:对应mask像素值，可添加多个标签如{"yiwu":1, "sanwu":2}
MASK_SUFFIX = ".png"  # mask图像后缀，也可以直接写".png"和原文件同名



def labelme_json2mask(json_path):
    """
    单文件转换：LabelMe JSON → Mask图像
    :param json_path: 单个json文件的路径
    """
    # 读取json文件
    with open(json_path, 'r', encoding='utf-8') as f:
        labelme_data = json.load(f)

    # 获取图像尺寸
    img_height = labelme_data["imageHeight"]
    img_width = labelme_data["imageWidth"]

    # 创建空白mask画布，背景像素值为0
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 遍历所有标注的shape
    for shape in labelme_data["shapes"]:
        label = shape["label"]
        shape_type = shape["shape_type"]
        points = shape["points"]

        # 只处理多边形标注（你的标签格式就是polygon，可兼容point/rectangle）
        if shape_type == "polygon" and label in LABEL_MAP:
            # 转换点坐标为cv2需要的整数格式
            pts = np.array(points, dtype=np.int32)
            # 填充多边形区域，像素值为LABEL_MAP中对应的数值
            cv2.fillPoly(mask, [pts], color=LABEL_MAP[label])

    # 生成mask保存路径
    mask_name = os.path.splitext(os.path.basename(json_path))[0] + MASK_SUFFIX
    mask_save_path = os.path.join(os.path.dirname(json_path), mask_name)

    # 保存mask图像
    cv2.imwrite(mask_save_path, mask)
    print(f"转换完成：{json_path} -> {mask_save_path}")


def batch_convert():
    """批量转换指定文件夹下所有json标签"""
    # 检查文件夹是否存在
    if not os.path.exists(JSON_DIR):
        print(f" 错误：文件夹 {JSON_DIR} 不存在！")
        return

    # 遍历文件夹内所有文件
    file_list = os.listdir(JSON_DIR)
    json_count = 0
    for file_name in file_list:
        # 只处理json后缀的文件
        if file_name.lower().endswith(".json"):
            json_path = os.path.join(JSON_DIR, file_name)
            labelme_json2mask(json_path)
            json_count += 1

    print(f"\n 批量转换完成！共处理 {json_count} 个JSON标签文件")


if __name__ == "__main__":
    batch_convert()