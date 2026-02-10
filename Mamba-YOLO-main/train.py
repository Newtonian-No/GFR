from ultralytics import YOLO

def train_custom_mamba():
    model = YOLO("yolo11_mamba.yaml") 

    model.train(
        data="data.yaml",      # 这里填数据集地址
        epochs=200,            
        imgsz=640,
        batch=16,
        device=0,              
        lr0=0.01,              
        project="Kidney_Mamba",
        name="spiral_vs_cross"
    )

if __name__ == "__main__":
    train_custom_mamba()