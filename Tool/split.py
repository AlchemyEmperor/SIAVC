
import os
import pandas as pd
import shutil

# 读取split.xlsx文件中第六列的字段内容
xlsx_file = "F:/SI9/split.xlsx"
df = pd.read_excel(xlsx_file)
column_name = df.columns[5]  # 第六列对应的字段名称

# 获取Train文件夹和Test文件夹的路径
train_folder = "F:/SI9/Train"
test_folder = "F:/SI9/Test"

# 遍历每一行的字段内容
for index, row in df.iterrows():
    keyword = row[column_name]

    # 遍历Train文件夹下的所有子文件夹
    for folder in os.listdir(train_folder):
        folder_path = os.path.join(train_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        # 在Train文件夹的每个子文件夹下搜索含有字段内容的文件并剪切到对应的Test子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and keyword in filename:
                # 确保对应的Test子文件夹存在，如果不存在则创建
                test_subfolder = os.path.join(test_folder, folder)
                os.makedirs(test_subfolder, exist_ok=True)

                # 剪切文件到Test子文件夹
                shutil.move(file_path, os.path.join(test_subfolder, filename))