# Smart Traffic Monitoring using Edge Computing
Real-Time Vehicle & Pedestrian Detection via Edge AI.
This project presents the design and implementation of a Smart Traffic Monitoring System using Edge Computing for real-time vehicle and pedestrian detection. The system processes video streams locally on an edge device using YOLO-based deep learning models, significantly reducing latency, bandwidth consumption, and privacy risks compared to traditional cloud-based approaches.
The project evaluates the effectiveness of Edge Computing for intelligent transportation systems by monitoring traffic conditions, detecting vehicles and pedestrians, measuring inference performance, and analyzing resource utilization. A custom object detection model trained using the IITM HETRA traffic dataset was also developed and converted into NCNN format for optimized deployment on Raspberry Pi devices.



## Features
- Real-Time Vehicle Detection
-	Real-Time Pedestrian Detection
-	YOLO-Based Edge AI Inference
-	Custom NCNN Traffic Detection Model
-	FPS Monitoring
-	CPU Usage Monitoring
-	Temperature Monitoring
-	Traffic Statistics Generation
-	Reduced Network Dependency
-	Enhanced Privacy Preservation


## Edge vs Cloud Comparison

| Parameter | Cloud Computing | Edge Computing |
|------------|------------------|-----------------|
| Latency | High | Low |
| Bandwidth Usage | High | Low |
| Privacy | Moderate | High |
| Internet Dependency | Required | Optional |
| Real-Time Response | Moderate | Excellent |
| Data Transmission | Continuous | Minimal |
| Infrastructure Cost | High | Lower |
## Installation
Provide clear steps to set up the environment locally:
```bash
git clone https://github.com/rakesh001n/edge-ai-detection-for-traffic-signal
pip instal ultralytics==8.3.187
pip install psutil==7.0.0
pip install opencv-python==4.10.0.84
```

## Usage
To train the custom model:
- Download this dataset [https://www.kaggle.com/datasets/deepak242424/iitmhetra](https://www.kaggle.com/datasets/deepak242424/iitmhetra)
- Rename the folder as Dataset_YOLO
- also rename the subfolders to ```images/train``` ,``` images/val``` as respectively for the train and validation dataset folders.
```bash
python3 custom_dataset.py
```
### Once the ```ncnn``` file is exported, run the following code
```bash
python3 custom_model.py
```

### To run the pre-built yolo model:
edit line 88 at ```custom_model.py``` file to the below code :
```code 
model = YOLO("yolo11n.pt", task='detect')
```



## License
Distributed under the MIT License. See `LICENSE` for more information.
