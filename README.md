# Smart Traffic Monitoring using Edge Computing

Real-Time Vehicle & Pedestrian Detection via Edge AI

This project presents the design and implementation of a Smart Traffic Monitoring System using Edge Computing for real-time vehicle and pedestrian detection. The system processes video streams locally on an edge device using YOLO-based deep learning models, significantly reducing latency, bandwidth consumption, and privacy risks compared to traditional cloud-based approaches.

The project evaluates the effectiveness of Edge Computing for intelligent transportation systems by monitoring traffic conditions, detecting vehicles and pedestrians, measuring inference performance, and analysing resource utilisation. A custom object detection model trained using the IITM HETRA traffic dataset was also developed and converted into NCNN format for optimised deployment on Raspberry Pi devices.

---

## Features

- Real-Time Vehicle Detection
- Real-Time Pedestrian Detection
- YOLO-Based Edge AI Inference
- Custom NCNN Traffic Detection Model
- FPS Monitoring
- CPU Usage Monitoring
- Temperature Monitoring
- Traffic Statistics Generation
- Reduced Network Dependency
- Enhanced Privacy Preservation

---

## Edge vs Cloud Comparison

| **Parameter**        | **Cloud Computing** | **Edge Computing** |
| -------------------- | ------------------- | ------------------ |
| Latency              | High                | Low                |
| Bandwidth Usage      | High                | Low                |
| Privacy              | Moderate            | High               |
| Internet Dependency  | Required            | Optional           |
| Real-Time Response   | Moderate            | Excellent          |
| Data Transmission    | Continuous          | Minimal            |
| Infrastructure Cost  | High                | Lower              |

---

## Prerequisites

Before starting, ensure you have the following hardware and software ready.

### Hardware Requirements

- Raspberry Pi 4 (4GB RAM recommended) or Raspberry Pi 5
- MicroSD card (32GB or larger, Class 10)
- USB Camera or Raspberry Pi Camera Module v2
- Power supply (5V 3A USB-C for Pi 4)
- Keyboard, mouse, and monitor (for initial setup) or SSH access

### Software Requirements

- Raspberry Pi OS (64-bit, Bullseye or Bookworm) — installed on the Pi
- Python 3.9 or later
- Internet connection (for initial setup and package installation)

---

## Raspberry Pi Setup and Configuration

### Step 1 — Flash Raspberry Pi OS

- Download the [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your PC.
- Insert the MicroSD card into your PC.
- Open Raspberry Pi Imager, select **Raspberry Pi OS (64-bit)**, choose your SD card, and click **Write**.
- Once flashing completes, insert the SD card into the Raspberry Pi and power it on.

### Step 2 — Initial OS Setup

- On first boot, follow the on-screen setup wizard to set your language, timezone, and Wi-Fi credentials.
- Enable SSH (optional, for headless access):

```bash
sudo raspi-config
# Navigate to: Interface Options → SSH → Enable
```

- Update the system:

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 3 — Install Python and System Dependencies

```bash
sudo apt install -y python3 python3-pip python3-venv git libgl1 libglib2.0-0
```

### Step 4 — Enable the Camera (if using Pi Camera Module)

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

sudo reboot
```

Verify the camera is detected after reboot:

```bash
libcamera-hello
```

---

## Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/rakesh001n/edge-ai-detection-for-traffic-signal
cd edge-ai-detection-for-traffic-signal
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python Dependencies

```bash
pip install ultralytics==8.3.187
pip install psutil==7.0.0
pip install opencv-python==4.10.0.84
```

> **Note:** Installation may take several minutes on Raspberry Pi due to limited processing power. Ensure stable internet connectivity during this step.

---

## Configuration

### Camera Source

By default, the system uses the first available camera (`/dev/video0`). To use a different source (e.g., a video file for testing), edit `main.py` and update the video source variable at the top of the file.

### Model Selection

The project supports two model options:

**Option A — Pre-built YOLO model (recommended for quick start):**

Edit line 88 of `custom_model.py`:

```python
model = YOLO("yolo11n.pt", task='detect')
```

**Option B — Custom NCNN model (optimised for Raspberry Pi):**

Edit line 88 of `custom_model.py`:

```python
model = YOLO(model_path, task='detect')
```

where `model_path` points to your exported NCNN model directory (see Training section below).

---

## Running the Project

### Using the Pre-built YOLO Model

After completing installation and configuration:

```bash
python3 main.py
```

You should see a live video window with bounding boxes drawn around detected vehicles and pedestrians, along with an overlay showing FPS, CPU usage, and temperature.

### Using the Custom NCNN Model

Ensure the NCNN model files are present in the `yolo11n_ncnn_model/` or `best_ncnn_model/` directory, then run:

```bash
python3 custom_model.py
```

---

## Training the Custom Model (Optional)

Follow these steps only if you want to retrain the model on the IITM HETRA dataset.

### Step 1 — Download the Dataset

Download the dataset from Kaggle: [https://www.kaggle.com/datasets/deepak242424/iitmhetra](https://www.kaggle.com/datasets/deepak242424/iitmhetra)

### Step 2 — Organise the Dataset

Rename and structure the downloaded folder as follows:

```
Dataset_YOLO/
├── images/
│   ├── train/      ← training images
│   └── val/        ← validation images
```

Place `Dataset_YOLO/` in the root of the cloned repository.

### Step 3 — Generate the YAML Configuration

```bash
python3 generate_yaml.py
```

This creates `data.yaml` which points to the dataset paths and class definitions.

### Step 4 — Train the Model

```bash
python3 custom_dataset.py
```

Training will run for the configured number of epochs and save the best weights to the `best_ncnn_model/` directory.

### Step 5 — Export to NCNN Format

The training script automatically exports the model to NCNN format after training completes. Once exported, switch `custom_model.py` to use the NCNN model path as described in the Configuration section.

---

## Project File Structure

```
edge-ai-detection-for-traffic-signal/
├── best_ncnn_model/        # Custom trained NCNN model files
├── results/                # Output images and training result graphs
├── yolo11n_ncnn_model/     # Pre-converted YOLO NCNN model files
├── custom_dataset.py       # Script to train custom model on IITM HETRA dataset
├── custom_model.py         # Main detection script using the custom/NCNN model
├── data.yaml               # YOLO dataset configuration file
├── generate_yaml.py        # Script to auto-generate data.yaml
├── label.py                # Utility script for labelling and annotation
├── main.py                 # Entry point for running detection with pre-built model
├── output.jpg              # Sample output image
├── test_ncnn.py            # Script to test the NCNN model independently
├── traffic_efficiency_log.csv  # Log file capturing traffic metrics over time
└── README.md
```

---

## Expected Outputs

When the system is running correctly, you should observe:

- A live video window showing detected vehicles and pedestrians highlighted with bounding boxes and class labels
- An on-screen overlay displaying FPS, CPU usage percentage, and device temperature
- Periodic entries written to `traffic_efficiency_log.csv` recording detection counts and system metrics

---

## Validation Steps

To confirm the system is working correctly after setup:

- **Camera check** — Run `libcamera-hello` (Pi Camera) or `python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` (USB camera). It should return `True`.
- **Dependency check** — Run `python3 -c "import ultralytics, cv2, psutil; print('All dependencies OK')"`. No errors should appear.
- **Model load check** — Run `python3 test_ncnn.py` to verify the NCNN model loads and runs inference without errors.
- **Live detection** — Run `python3 main.py` and confirm bounding boxes appear around vehicles and pedestrians in the video feed.
- **Logging check** — After running for at least 30 seconds, open `traffic_efficiency_log.csv` and verify rows are being written.

---

## Troubleshooting

### Camera not detected

Run `ls /dev/video*` to check available camera devices. If nothing is listed, reconnect the camera and reboot. For Pi Camera, re-enable it via `raspi-config`.

### `ImportError` for `cv2` or `ultralytics`

Ensure the virtual environment is activated (`source venv/bin/activate`) and re-run the pip install commands.

### Low FPS on Raspberry Pi

- Use the NCNN model (`best_ncnn_model/`) instead of the `.pt` model — it is significantly faster on ARM hardware.
- Reduce input resolution in `main.py` by lowering the frame resize dimensions.
- Close all other running applications to free CPU resources.

### High CPU temperature / thermal throttling

Attach a heatsink and/or fan to the Raspberry Pi. You can monitor temperature in real time with:

```bash
watch -n 1 vcgencmd measure_temp
```

### NCNN model not found

Confirm the model files (`.param` and `.bin`) exist inside the `best_ncnn_model/` or `yolo11n_ncnn_model/` directory. If missing, re-run the training and export steps.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
