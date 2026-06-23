# Project Title
A brief, clear description of what your project does and why it exists.

## Features
* Feature item one
* Feature item two

## Installation
Provide clear steps to set up the environment locally:
```bash
git clone https://github.com/rakesh001n/edge-ai-detection-for-traffic-signal
pip instal ultralytics==8.3.187
pip install psutil==7.0.0
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
