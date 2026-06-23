import xml.etree.ElementTree as ET
import os
import glob
import shutil

# 1. Match your XML names EXACTLY
classes = ["Car", "Person", "Bus", "Truck", "Motorcycle"]

def convert_coordinates(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def organize_dataset(src_img_dir, src_xml_dir, output_root, split_name):
    """
    Creates parallel images/split and labels/split folders.
    split_name: 'train' or 'val'
    """
    dest_img_dir = os.path.join(output_root, "images", split_name)
    dest_lbl_dir = os.path.join(output_root, "labels", split_name)
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    xml_files = glob.glob(os.path.join(src_xml_dir, "*.xml"))
    print(f"Organizing {len(xml_files)} files into {split_name}...")

    for xml_path in xml_files:
        file_base = os.path.splitext(os.path.basename(xml_path))[0]
        img_path = os.path.join(src_img_dir, f"{file_base}.jpg")

        if not os.path.exists(img_path):
            continue

        # 1. Convert XML and save to labels folder
        txt_path = os.path.join(dest_lbl_dir, f"{file_base}.txt")
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            w, h = int(size.find('width').text), int(size.find('height').text)

            with open(txt_path, 'w') as f:
                for obj in root.iter('object'):
                    name = obj.find('name').text
                    if name not in classes: continue
                    idx = classes.index(name)
                    box = obj.find('bndbox')
                    b = (float(box.find('xmin').text), float(box.find('xmax').text), 
                         float(box.find('ymin').text), float(box.find('ymax').text))
                    bb = convert_coordinates((w, h), b)
                    f.write(f"{idx} {' '.join([f'{a:.6f}' for a in bb])}\n")
            
            # 2. Copy image to images folder
            shutil.copy(img_path, os.path.join(dest_img_dir, f"{file_base}.jpg"))
        except Exception as e:
            print(f"Error processing {file_base}: {e}")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    dataset_final = r"d:\\Projects\\Traffic Paper\Dataset_YOLO"
    
    # Process Train
    organize_dataset(
        src_img_dir=r"d:\\Projects\\Traffic Paper\Dataset\\images",
        src_xml_dir=r"d:\Projects\\Traffic Paper\Dataset\\xmls",
        output_root=dataset_final,
        split_name="train"
    )
    
    # Process Val
    organize_dataset(
        src_img_dir=r"d:\\Projects\\Traffic Paper\Dataset\\images_test",
        src_xml_dir=r"d:\\Projects\\Traffic Paper\Dataset\\xmls_test",
        output_root=dataset_final,
        split_name="val"
    )
    print(f"Done! Dataset ready at {dataset_final}")