import json 
import os
import shutil
import cv2 


def del_no_match_image(label_folder:str, img_folder:str):
    try:

        # 이미지 디렉토리에서 모든 파일 이름을 가져옴
        image_files = os.listdir(img_folder)

        # txt 디렉토리에서 모든 파일 이름을 가져옴
        txt_files = os.listdir(label_folder)

        # 이미지 파일 이름과 같은 이름의 txt 파일이 있는지 확인하고, 없으면 삭제
        for txt_file in txt_files:
            # txt 파일의 확장자를 제외한 이름을 가져옴
            base_name = os.path.splitext(txt_file)[0]
            # 같은 이름의 이미지 파일이 있는지 확인
            image_file = base_name + '.jpg'  
            if image_file not in image_files:
                # 같은 이름의 이미지 파일이 없다면, txt 파일 삭제
                os.remove(os.path.join(label_folder, txt_file))
                print("delete", txt_file)
                
    except Exception as e:
        print(e)


def convert_yolo_label(xyxy:list, img_size_hw:list) -> list:
    '''
    Labeling 된 BBOX의 x1, y1, x2, y2 좌표를 YoloStyle 로 변환 하는 코드
    xywh 라면 밑에 w와 h 계산할 필요 없이 바로 넣는다.
    '''
    
    (x1, y1, x2, y2) = xyxy
    img_h, img_w  = img_size_hw
    ##
    dw = 1./img_w
    dh = 1./img_h
    
    #xyxy2yolo_xywh
    x = float(x1)
    y = float(y1)
    w = float(x2)-float(x1)     #xyxy2xywh
    h = float(y2)-float(y1)

    x = (x + x + w)/2.0 
    y = (y + y + h)/2.0
    w = w
    h = h

    x = round(x*dw, 6)
    w = round(w*dw, 6)
    y = round(y*dh, 6)
    h = round(h*dh, 6)

    return [x, y, w, h]


def run(img_file_path:str, label_file_path:str, destination_path:str):
    try:
        label_files = [file for file in os.listdir(label_file_path) if file.endswith(".txt")]
        img_files = [file for file in os.listdir(img_file_path) if file.endswith(".jpg")]
        if not len(label_files) == len(img_files):
            print("nomatch")
            del_no_match_image(label_file_path, img_file_path)

        # assert len(label_files) == len(img_files), "Number of label files and image files should be same."

        for label_file in label_files:
            label_path = os.path.join(label_file_path, label_file)
            img_path = os.path.join(img_file_path, label_file.replace('.txt', '.jpg'))
            
            img = cv2.imread(img_path)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            #print(img.shape[:2])
            yolo_labels = []
            for line in lines:
                cls, x1, y1, x2, y2 = line.strip().split()
                yolo_style_bbox = convert_yolo_label([x1, y1, x2, y2], img.shape[:2])
                yolo_labels.append([cls] + yolo_style_bbox)

            # Write yolo labels to a new file in destination path
            with open(os.path.join(destination_path, label_file), 'w') as f:
                for yolo_label in yolo_labels:
                    #print(f)
                    f.write(' '.join(map(str, yolo_label)) + '\n')


    except Exception as e:
        print(e)