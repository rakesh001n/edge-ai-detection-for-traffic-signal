import cv2
import os
from ultralytics import YOLO

def test_ncnn_detection(model_path, image_path, conf_threshold=0.25):
    """
    Tests an NCNN model and displays results with class names.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model folder '{model_path}' not found.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    print(f"Loading NCNN model from: {model_path}...")
    # Load the model - ensure task is set correctly
    model = YOLO(model_path, task='detect')

    # Run inference
    print(f"Running inference on: {image_path}...")
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    # Get class names dictionary from the model
    # NCNN models usually bundle this in metadata.yaml inside the folder
    class_names = model.names
    print(f"Detected Classes in Model: {class_names}")

    # Load image for manual drawing (to ensure we see exactly what's happening)
    img = cv2.imread(image_path)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class ID
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = class_names.get(cls_id, f"Unknown({cls_id})")
            
            # Draw rectangle
            color = (0, 255, 0) # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            
            # Put text
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            print(f"Found: {label} with {conf:.2f} confidence at [{x1}, {y1}, {x2}, {y2}]")

    # Save and show result
    output_name = "output.jpg"
    cv2.imwrite(output_name, img)
    print(f"\nResult saved as {output_name}")
    
    # Note: cv2.imshow might not work in all environments (like SSH/headless)
    # But if you are on a local machine with a screen:
    # cv2.imshow("NCNN Test", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # CHANGE THESE PATHS
    MODEL_FOLDER = 'best_ncnn_model' # The folder containing model.ncnn.param etc.
    TEST_IMAGE = 'Dataset\\images\\frame_9.jpg'          # An image you want to test
    
    test_ncnn_detection(MODEL_FOLDER, TEST_IMAGE)
