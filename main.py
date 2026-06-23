import cv2
from ultralytics import YOLO

def main():
    # OPTION A: YOLO11n - Highly efficient, very stable on RPi 5
    model = YOLO('yolo11n.pt') 
    
    # OPTION B: YOLO26n - The newest (2026) edge-optimized model
    # To use this, run: pip install -U ultralytics
    # model = YOLO('yolo26n.pt') 

    # RPi 5 Optimization: Lower resolution for higher FPS
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    line_y = 320 
    vehicle_count = 0
    prev_centers = {}

    print(f"Starting {model.model_name} Monitor... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # .track() uses BoT-SORT or ByteTrack automatically
        # classes: 2:car, 3:motorcycle, 5:bus, 7:truck
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if track_id in prev_centers:
                    prev_cy = prev_centers[track_id]
                    # Logic: If center moved from above line to below line
                    if prev_cy < line_y <= cy:
                        vehicle_count += 1
                
                prev_centers[track_id] = cy

                # Visualization
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # UI Overlay
        cv2.line(frame, (0, line_y), (640, line_y), (255, 0, 0), 3)
        cv2.putText(frame, f"VEHICLE COUNT: {vehicle_count}", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("RPi 5 Edge Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()