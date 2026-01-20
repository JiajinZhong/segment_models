import os
import shutil

# ====================== 【必改配置 - 只改这4行！】 ======================
IMAGE_DIR = r"E:\A_TCL_data\M3B\M3B_PCBI\segment_datas\yiwu"  # 你的【图像】文件夹路径
LABEL_DIR = r"E:\A_TCL_data\M3B\M3B_PCBI\segment_datas\yiwu_labels"  # 你的【标签json】文件夹路径
SAVE_ROOT = r"E:\A_TCL_data\M3B\M3B_PCBI\segment_datas\selected_data"  # 筛选后的文件保存根目录（自动创建）
IMAGE_SUFFIX = [".bmp", ".png", ".jpg", ".jpeg"]  # 你的图像格式，你的是.bmp，不用改



def get_file_basename(file_path):
    """获取文件的纯名称（去掉路径和后缀）"""
    return os.path.splitext(os.path.basename(file_path))[0]


def copy_file(src_path, dst_path):
    """安全拷贝文件，自动创建文件夹"""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)  # copy2会保留文件原属性


def select_paired_data():
    # 收集所有标签文件 存入集合用于快速匹配
    label_basename_set = set()
    for label_file in os.listdir(LABEL_DIR):
        if label_file.lower().endswith(".json"):  # 标签是json文件
            label_basename = get_file_basename(label_file)
            label_basename_set.add(label_basename)
    print(f"共读取到 {len(label_basename_set)} 个标签文件")

    # 创建保存路径
    save_img_dir = os.path.join(SAVE_ROOT, "selected_images")
    save_lab_dir = os.path.join(SAVE_ROOT, "selected_labels")

    # 遍历图像文件夹，筛选有对应标签的图像+标签
    matched_count = 0
    for img_file in os.listdir(IMAGE_DIR):
        # 筛选指定格式的图像文件
        img_suffix = os.path.splitext(img_file)[1].lower()
        if img_suffix not in IMAGE_SUFFIX:
            continue

        img_basename = get_file_basename(img_file)
        # 核心匹配逻辑：图像纯名称 在 标签纯名称集合中 → 匹配成功
        if img_basename in label_basename_set:
            # 拼接源文件路径
            src_img = os.path.join(IMAGE_DIR, img_file)
            src_lab = os.path.join(LABEL_DIR, img_basename + ".json")

            # 拼接目标文件路径
            dst_img = os.path.join(save_img_dir, img_file)
            dst_lab = os.path.join(save_lab_dir, img_basename + ".json")

            # 拷贝文件
            copy_file(src_img, dst_img)
            copy_file(src_lab, dst_lab)

            matched_count += 1
            print(f" 匹配成功并拷贝：{img_file} + {img_basename}.json")

    # 4. 统计结果
    print("=" * 50)
    print(f" 筛选完成！共筛选出 {matched_count} 对【图像+标签】文件")
    print(f" 筛选后的图像路径：{save_img_dir}")
    print(f" 筛选后的标签路径：{save_lab_dir}")


if __name__ == "__main__":
    select_paired_data()