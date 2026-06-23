import cv2
from ultralytics import YOLO
import numpy as np
import time
import psutil
import os
import csv
import platform
from datetime import datetime

# --- SYSTEM UTILITIES ---
def get_system_temperature():
    try:
        if platform.system() == "Linux":
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return int(f.read()) / 1000.0
        else:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    return entries[0].current
            return 0.0
    except:
        return 0.0

def get_system_voltage():
    try:
        if platform.system() == "Linux":
            res = os.popen('vcgencmd measure_volts core').readline()
            return float(res.replace("volt=","").replace("V\n",""))
        return 0.0
    except:
        return 0.0

# --- PRIVACY UTILITIES ---
def apply_custom_blur(frame, x1, y1, x2, y2, region_type="face"):
    """Applies Gaussian Blur to specific regions (Face for humans, Plate for vehicles)."""
    try:
        regions = []
        if region_type == "face":
            # Blur top 35% and bottom 35%
            regions.append((int(x1), int(y1), int(x2), int(y1 + (y2 - y1) * 0.35)))
            regions.append((int(x1), int(y2 - (y2 - y1) * 0.35), int(x2), int(y2)))
        else:
            # Blur the bottom 40% of the bounding box (where the license plate usually is)
            regions.append((int(x1), int(y2 - (y2 - y1) * 0.4), int(x2), int(y2)))

        for rx1, ry1, rx2, ry2 in regions:
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(frame.shape[1], rx2), min(frame.shape[0], ry2)
            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2]
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                frame[ry1:ry2, rx1:rx2] = blurred_roi
    except:
        pass

# --- MAIN ENGINE ---
def main():
    # 1. Setup CSV Logging
    log_file = "traffic_efficiency_log.csv"
    file_exists = os.path.isfile(log_file)
   
    csv_f = open(log_file, mode='a', newline='')
    log_writer = csv.writer(csv_f)
   
    if not file_exists:
        log_writer.writerow([
            "Timestamp", "Inference_ms", "E2E_Delay_ms",
            "FPS", "CPU_Usage_Percent", "RAM_Usage_Percent",
            "Temp_C", "CPU_Freq_MHz", "Voltage_V",
            "Vehicle_Count", "People/bike count"
        ])

    # 2. Model Loading
    model_path = 'best_ncnn_model'
    #if not os.path.exists(model_path) or platform.system() == "Windows":
    #    print("Running standard .pt model (NCNN not detected or Windows environment)...")
    #    if os.path.exists("./results\traffic_paper\\v1_edge_model\\weights\\best.pt"):
    #        print("best.pt loaded")
    #        model = YOLO("./results\traffic_paper\\v1_edge_model\\weights\\best.pt")
#
    #    else:
    #        print("yolo11n.pt loaded")
    #        model = YOLO("yolo11n.pt")
    #else:
    print("Loading optimized NCNN model...")
    model = YOLO("yolo11n.pt", task='detect')

    # Dynamically get class IDs from the model's names to avoid hardcoding errors
    VEHICLE_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck', 'vehicle', 'auto']
    PEDESTRIAN_CLASS_NAMES = ['person', 'pedestrian', 'human']

    VEHICLE_CLASSES = [k for k, v in model.names.items() if str(v).lower().strip() in VEHICLE_CLASS_NAMES]
    PEDESTRIAN_CLASSES = [k for k, v in model.names.items() if str(v).lower().strip() in PEDESTRIAN_CLASS_NAMES]

    print("--- Model Class Information ---")
    print(f"All class names: {model.names}")
    print(f"Using PEDESTRIAN class IDs: {PEDESTRIAN_CLASSES}")
    print(f"Using VEHICLE class IDs: {VEHICLE_CLASSES}")
    print("-----------------------------")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   
    frame_count = 0
    start_time = time.time()

    print(f"Logging data to {log_file}...")
    print("Press 'q' to stop.")

    try:
        while cap.isOpened():
            e2e_start = time.time()
            success, frame = cap.read()
            if not success: break

            # Inference
            inf_start = time.time()
            results = model.track(frame, persist=True, imgsz=320, verbose=False)
            inf_end = time.time()
            inference_latency = (inf_end - inf_start) * 1000

            # Reset counts for current frame
            current_vehicles = 0
            current_pedestrians = 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = box
                   
                    # LOGIC: Differentiate between People and Vehicles
                    if cls in PEDESTRIAN_CLASSES:
                        current_pedestrians += 1
                        apply_custom_blur(frame, x1, y1, x2, y2, "face")
                        # (Optional) Use a different color for person bounding boxes
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                   
                    elif cls in VEHICLE_CLASSES:
                        current_vehicles += 1
                        apply_custom_blur(frame, x1, y1, x2, y2, "plate")
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Efficiency Metrics
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            sys_temp = get_system_temperature()
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            voltage = get_system_voltage()

            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            e2e_delay = (time.time() - e2e_start) * 1000

            # Log to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_writer.writerow([
                timestamp, round(inference_latency, 2), round(e2e_delay, 2),
                round(fps, 2), cpu_usage, ram_usage,
                sys_temp, round(cpu_freq, 1), voltage,
                current_vehicles, current_pedestrians
            ])
           
            if frame_count % 10 == 0: csv_f.flush()

            # Display GUI
            overlay_color = (0, 255, 255)
            metrics = [
                f"FPS: {fps:.1f} | E2E: {e2e_delay:.1f}ms",
                f"CPU: {cpu_usage}% | Temp: {sys_temp:.1f}C",
                f"Vehicles: {current_vehicles} | People: {current_pedestrians}"
            ]

            for i, text in enumerate(metrics):
                cv2.putText(frame, text, (10, 30 + (i * 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)

            cv2.imshow("Smart Traffic Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_f.close()
        print(f"Detailed logs saved to {log_file}")

if __name__ == "__main__":
    main()