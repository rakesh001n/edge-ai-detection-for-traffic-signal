from ultralytics import YOLO
import torch
import os

def main():
    # 1. Load the latest YOLO11 Nano model
    model = YOLO('yolo11n.pt')

    # 2. Check for CUDA (GPU) availability
    # If a GPU is found, use it (0). Otherwise, use 'cpu'.
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 3. Train the model
    results = model.train(
        data='data.yaml',
        epochs=100,         # Standard for research papers
        imgsz=640,          # Training resolution
        batch=16,           # Reduce this to 8 or 4 if CPU training is too slow/crashes
        optimizer='AdamW',  # Better for edge-device quantization
        project='traffic_research',
        name='yolo11_traffic_v1',
        device=device       # Automatically switches between GPU and CPU
    )

    print("Training complete. Validating results...")
    metrics = model.val()

    # 4. Export for Raspberry Pi 5
    # Exporting to NCNN with INT8 quantization for maximum edge speed
    print(metrics)
    print("Exporting model to NCNN format for Raspberry Pi 5...")
    # Note: Exporting on CPU is perfectly fine
    model.export(format='ncnn', int8=True, imgsz=320)

if __name__ == "__main__":
    main()