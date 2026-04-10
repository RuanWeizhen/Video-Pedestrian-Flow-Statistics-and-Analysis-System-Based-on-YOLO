import cv2  # 导入 OpenCV 库，用于处理视频和图像
import os   # 导入 os 库，用于文件和路径操作
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条（如果没有安装，请先执行 pip install tqdm）

# ==================== 配置参数（请根据需要修改） ====================
# 输入视频文件夹路径
video_folder = r"E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\make_dataset\videos"
# 输出图片文件夹路径
output_folder = r"E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\make_dataset\images"
# 每隔多少帧提取一张图片（例如 30 表示每 30 帧保存一帧）
frame_interval = 30
# ===================================================================

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建输出文件夹：{output_folder}")

# 获取所有 .mp4 文件（大小写不敏感，可匹配 .MP4 等）
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]

# 如果没有找到视频文件，提示并退出
if not video_files:
    print("在指定文件夹中未找到任何 .mp4 文件，请检查路径。")
    exit()

print(f"共找到 {len(video_files)} 个视频文件，开始处理...")

# 遍历每个视频文件
for video_file in video_files:
    # 获取视频文件名（不含扩展名），用于命名图片
    video_name = os.path.splitext(video_file)[0]  # 例如 "001"
    # 视频完整路径
    video_path = os.path.join(video_folder, video_file)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}，跳过处理。")
        continue

    # 获取视频的总帧数（用于进度条显示）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 如果获取不到总帧数（例如某些格式），则设为 None，进度条将不显示总数
    if total_frames <= 0:
        total_frames = None

    # 初始化帧计数器（从0开始计数）和图片序号（从1开始）
    frame_count = 0
    img_index = 1

    print(f"正在处理视频：{video_file}")

    # 使用 tqdm 显示进度条，desc 为当前视频名称，total 为总帧数（如果已知）
    with tqdm(total=total_frames, desc=video_name, unit="帧") as pbar:
        while True:
            # 读取一帧，ret 为布尔值表示是否成功读取，frame 为图像数据
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取完毕或出错，退出循环

            # 判断当前帧是否满足间隔条件
            if frame_count % frame_interval == 0:
                # 构建图片文件名：视频名_5位序号.jpg，例如 001_00001.jpg
                img_name = f"{video_name}_{img_index:05d}.jpg"
                # 构建图片完整保存路径
                img_path = os.path.join(output_folder, img_name)
                # 保存图片
                cv2.imwrite(img_path, frame)
                # 图片序号加1
                img_index += 1

            # 帧计数器加1
            frame_count += 1
            # 更新进度条
            pbar.update(1)

    # 释放视频资源
    cap.release()
    print(f"视频 {video_file} 处理完毕，共提取 {img_index - 1} 张图片。")

print("所有视频处理完成！")