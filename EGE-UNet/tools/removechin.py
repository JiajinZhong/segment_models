# import os
# import re
#
# # ====================== 【仅需修改这2处配置，其他不用动】 ======================
# TARGET_DIR = r"E:\A_TCL_data\M3B\M3B_PCBI\segment_datas\yiwu"  # 你的图像文件夹路径
# IMAGE_SUFFIX = [".bmp", ".png", ".jpg", ".jpeg", ".tif"]  # 需要处理的图像后缀，你的是.bmp
#
#
# # ================================================================================
#
# def remove_chinese_from_filename(file_path):
#     """移除单个文件名称中的所有中文字符"""
#     file_dir, file_name = os.path.split(file_path)
#     name, suffix = os.path.splitext(file_name)
#     suffix = suffix.lower()
#
#     # 过滤非图像文件，跳过
#     if suffix not in IMAGE_SUFFIX:
#         return False
#
#     # 正则表达式：匹配【所有中文字符】(\u4e00-\u9fff 是中文unicode编码范围)
#     pattern = re.compile(r'[\u4e00-\u9fff]+')
#     new_name = pattern.sub('', name)  # 把匹配到的中文替换为空字符
#
#     # 如果新名字和原名字一样（无中文），跳过
#     if new_name == name:
#         print(f"➡️ 跳过：{file_name} (无中文字符)")
#         return False
#
#     # 拼接新文件完整路径
#     new_file_name = new_name + suffix
#     new_file_path = os.path.join(file_dir, new_file_name)
#
#     # 处理重名：如果新文件名已存在，添加数字后缀
#     if os.path.exists(new_file_path):
#         count = 1
#         while True:
#             temp_name = f"{new_name}_{count}{suffix}"
#             temp_path = os.path.join(file_dir, temp_name)
#             if not os.path.exists(temp_path):
#                 new_file_path = temp_path
#                 new_file_name = temp_name
#                 break
#             count += 1
#
#     # 执行重命名
#     os.rename(file_path, new_file_path)
#     print(f"✅ 重命名成功：{file_name} --> {new_file_name}")
#     return True
#
#
# def batch_process(recursive=False):
#     """批量处理文件夹内文件，recursive=True 处理子文件夹，False只处理当前文件夹"""
#     if not os.path.exists(TARGET_DIR):
#         print(f"❌ 错误：文件夹路径 {TARGET_DIR} 不存在！")
#         return
#
#     process_count = 0
#     # 遍历文件夹
#     if recursive:
#         for root, dirs, files in os.walk(TARGET_DIR):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 if remove_chinese_from_filename(file_path):
#                     process_count += 1
#     else:
#         for file in os.listdir(TARGET_DIR):
#             file_path = os.path.join(TARGET_DIR, file)
#             if os.path.isfile(file_path):  # 只处理文件，不处理文件夹
#                 if remove_chinese_from_filename(file_path):
#                     process_count += 1
#
#     print("=" * 60)
#     print(f"🎉 批量处理完成！共修改了 {process_count} 个图像文件的名称")
#
#
# if __name__ == "__main__":
#     # 如需处理子文件夹，把下面的 False 改成 True 即可
#     batch_process(recursive=False)

import os
import re


LABEL_DIR = r"E:\A_TCL_data\M3B\M3B_PCBI\segment_datas\yiwu_labels"  # 标签文件夹路径
LABEL_SUFFIX = [".json", ".png"]  # json标签 / mask标签都支持



def remove_chinese_label(file_path):
    file_dir, file_name = os.path.split(file_path)
    name, suffix = os.path.splitext(file_name)
    suffix = suffix.lower()

    if suffix not in LABEL_SUFFIX:
        return False

    pattern = re.compile(r'[\u4e00-\u9fff]+')
    new_name = pattern.sub('', name)
    if new_name == name:
        print(f" 跳过：{file_name} (无中文字符)")
        return False

    new_file_name = new_name + suffix
    new_file_path = os.path.join(file_dir, new_file_name)

    if os.path.exists(new_file_path):
        count = 1
        while True:
            temp_name = f"{new_name}_{count}{suffix}"
            temp_path = os.path.join(file_dir, temp_name)
            if not os.path.exists(temp_path):
                new_file_path = temp_path
                new_file_name = temp_name
                break
            count += 1

    os.rename(file_path, new_file_path)
    print(f" 标签重命名成功：{file_name} --> {new_file_name}")
    return True


def batch_label_process(recursive=False):
    if not os.path.exists(LABEL_DIR):
        print(f" 错误：标签文件夹 {LABEL_DIR} 不存在！")
        return

    process_count = 0
    if recursive:
        for root, dirs, files in os.walk(LABEL_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                if remove_chinese_label(file_path):
                    process_count += 1
    else:
        for file in os.listdir(LABEL_DIR):
            file_path = os.path.join(LABEL_DIR, file)
            if os.path.isfile(file_path):
                if remove_chinese_label(file_path):
                    process_count += 1

    print("=" * 60)
    print(f" 标签文件处理完成！共修改 {process_count} 个标签文件")


if __name__ == "__main__":
    batch_label_process(recursive=False)