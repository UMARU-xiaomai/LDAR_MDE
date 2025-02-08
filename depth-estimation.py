import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def load_model(model_type):
    """加载本地MiDaS模型"""
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transform, device

def normalize_depth_map(depth_map):
    """将深度图归一化到0-255范围，便于可视化"""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_depth = 255 * (depth_map - depth_min) / (depth_max - depth_min)
    return normalized_depth.astype(np.uint8)

def create_heatmap(depth_map):
    """将深度图转换为彩色热力图"""
    normalized_depth = normalize_depth_map(depth_map)
    heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
    return heatmap

def process_video(video_path, model, transform, device):
    """处理视频并实时显示结果"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1/fps if fps > 0 else 1/30  # 计算每帧应该的展示时间
    
    # 创建窗口
    cv2.namedWindow('Depth Visualization', cv2.WINDOW_NORMAL)
    
    center_depths = []
    timestamps = []
    frame_count = 0
    
    while cap.isOpened():
        start_time = time.time()  # 记录开始处理时间
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # 准备输入
        input_batch = transform(frame).to(device)
        
        # 预测深度
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # 获取中心点深度值
        h, w = depth_map.shape
        center_depth = depth_map[h//2, w//2]
        center_depths.append(center_depth)
        timestamps.append(frame_count / fps)
        
        # 创建热力图
        heatmap = create_heatmap(depth_map)
        
        # 创建显示图像
        # 调整frame和heatmap大小使其相同
        frame = cv2.resize(frame, (640, 480))
        heatmap = cv2.resize(heatmap, (640, 480))
        
        # 水平拼接原图和热力图
        combined_img = np.hstack((frame, heatmap))
        
        # 添加黑色底部区域显示深度值
        bottom_height = 50
        bottom = np.zeros((bottom_height, combined_img.shape[1], 3), dtype=np.uint8)
        
        # 在底部添加文字
        text = f'Center Depth: {center_depth:.2f}'
        cv2.putText(bottom, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        
        # 垂直拼接
        final_img = np.vstack((combined_img, bottom))
        
        # 显示结果
        cv2.imshow('Depth Visualization', final_img)
        
        # 计算需要等待的时间以匹配原视频帧率
        process_time = time.time() - start_time
        wait_time = max(1, int((frame_time - process_time) * 1000))
        
        # 检查是否按下'q'键退出
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    cv2.destroyAllWindows()
    return timestamps, center_depths

def plot_depth_curve(timestamps, depths):
    """绘制深度随时间变化的曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, depths)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Relative Depth')
    plt.title('Center Depth vs Time')
    plt.grid(True)
    plt.show()

def main(video_path, model_type):
    """主函数"""
    print("Loading model...")
    model, transform, device = load_model(model_type)
    
    print("Processing video...")
    timestamps, depths = process_video(video_path, model, transform, device)
    
    print("Plotting results...")
    plot_depth_curve(timestamps, depths)
    
    return timestamps, depths

if __name__ == "__main__":
    video_path = 'D:/ShimitsuKoi/LDAR_MDE/videos/N01091919.mp4'
    model_type = 'DPT_BEiT_B_384'
    main(video_path, model_type)