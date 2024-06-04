import os
import imageio
import pickle

# 定义文件夹路径
folder_path = r'F:\KeyPoint\Code\FixMatch-pytorch-master\dataset\Firesense_Test\Smoke'

# 初始化总帧数
total_frames = 0

# 获取.pkl文件列表
pkl_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

# 循环读取每个.pkl文件并累计帧数
for pkl_file in pkl_files:
    pkl_path = os.path.join(folder_path, pkl_file)

    # 打开.pkl文件并加载视频文件名（假设文件中保存的是字符串）
    with open(pkl_path, 'rb') as f:
        video_file = pickle.load(f)

    # 构建视频文件的完整路径
    # video_path = os.path.join(folder_path, video_file_name)

    # 使用imageio打开视频文件并获取帧数
    # video_reader = imageio.get_reader(video_path)
    # frames_in_video = len(video_reader)
    total_frames += video_file.shape[0]

    # 关闭video_reader
    # video_reader.close()

# 打印总帧数
print(f"Total frames in all videos: {total_frames}")

