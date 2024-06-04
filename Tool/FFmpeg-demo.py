import os
import subprocess

input_directory = r'E:\迅雷下载\安全事故数据集（7类）\训练集\漏雨'
output_directory = r'E:\迅雷下载\安全事故数据集（7类）\训练集\漏雨\output'

# 创建输出目录
os.makedirs(output_directory, exist_ok=True)

# 枚举目录中的所有视频文件
for filename in os.listdir(input_directory):
    if filename.endswith('.mp4'):  # 假设你处理的是MP4文件
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # 使用FFmpeg命令行将帧率更改为15帧
        command = f'ffmpeg -i "{input_path}" -r 15 "{output_path}"'

        try:
            subprocess.run(command, shell=True, check=True)
            print(f"已处理文件: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"处理文件 {filename} 时出错:", e)

print("所有视频处理完成。")

# # 视频文件路径
# video_path = "D:/1.mp4"
#
# # 切片起始时间
# start_time = "00:29:20"
#
# # 切片结束时间
# end_time = "00:31:30"
#
# # 切片文件路径
# output_path = "D:/output.mp4"
#
# # 构造FFmpeg命令行
# command = f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v copy -c:a copy {output_path}"
#
# # 调用FFmpeg命令行
# subprocess.call(command, shell=True)