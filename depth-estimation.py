import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import time

def process_video_depth(video_path, output_path):
    """
    使用DepthFM处理视频并输出中心点深度变化曲线
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
    """
    # 初始化DepthFM深度估计模型
    depth_estimator = pipeline('depth-estimation', model='LiheYoung/depth-fm-large')
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 存储中心点深度值
    center_depths = []
    frame_times = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # 深度估计
        depth_map = depth_estimator(image)['depth']
        
        # 获取深度图
        depth_np = np.array(depth_map)
        
        # 获取中心点深度值
        center_y = frame_height // 2
        center_x = frame_width // 2
        center_depth = depth_np[center_y, center_x]
        
        # 记录深度值和时间
        center_depths.append(center_depth)
        frame_times.append(frame_count / fps)
        
        # 可视化深度图
        depth_colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_np, alpha=50), 
            cv2.COLORMAP_JET
        )
        
        # 在深度图上标记中心点
        cv2.circle(depth_colored, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 写入输出视频
        out.write(depth_colored)
        
        frame_count += 1
        
    # 释放资源
    cap.release()
    out.release()
    
    # 绘制深度变化曲线
    plt.figure(figsize=(12, 6))
    plt.plot(frame_times, center_depths)
    plt.title('Center Point Depth Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth')
    plt.grid(True)
    plt.savefig('depth_curve.png')
    plt.close()
    
    return frame_times, center_depths

def visualize_results(frame_times, center_depths):
    """
    创建交互式深度变化曲线图
    """
    plt.figure(figsize=(12, 6))
    plt.plot(frame_times, center_depths, 'b-')
    plt.title('Depth Variation at Video Center')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Estimated Depth (meters)')
    plt.grid(True)
    plt.show()

# 使用示例
if __name__ == "__main__":
    video_path = "input_video.mp4"
    output_path = "output_depth.mp4"
    
    # 处理视频并获取数据
    times, depths = process_video_depth(video_path, output_path)
    
    # 显示结果
    visualize_results(times, depths)
