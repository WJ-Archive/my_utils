'''
압축 해제 명령어 
for file in ~/_traffic_sig/zip_data/Training/label*.tar; do
    tar -xvf "$file" -C ~/_traffic_sig/yolov5/dataset/labels_json/train/
done

for file in ~/_traffic_sig/zip_data/Training/raw*.tar; do
    tar -xvf "$file" -C ~/_traffic_sig/yolov5/dataset/images/train/
done

for file in ~/_traffic_sig/zip_data/Validation/label*.tar; do
    tar -xvf "$file" -C ~/_traffic_sig/yolov5/dataset/labels_json/val/
done

for file in ~/_traffic_sig/zip_data/Validation/raw*.tar; do
    tar -xvf "$file" -C ~/_traffic_sig/yolov5/dataset/images/val/
done
'''
# jpg 파일을 /dataset/images/train/ 디렉토리로 이동
#find /home/woojin/_traffic_sig/yolov5/dataset/aug_data/train -type f -name "*.jpg" -exec mv {} /home/woojin/_traffic_sig/yolov5/dataset/images/train \;

# txt 파일을 /dataset/labels/train/ 디렉토리로 이동
#find /home/woojin/_traffic_sig/yolov5/dataset/aug_data/train -type f -name "*.txt" -exec mv {} /home/woojin/_traffic_sig/yolov5/dataset/labels/train \;

'''
#데이터 전처리 과정
#1. JSON 데이터셋 읽어와서 매칭되는 이미지와, 라벨링 파일 나누기
#2. 읽어온 데이터셋에서 필요한 데이터셋만 뽑아내서 txt 파일로 변환 *cls x1 y1 x2 y2 형식으로 기록 (labels_txt 에 저장.) 
#3. 데이터 Augmentation, Sampling, Delete 등을 통해 data Balance 잡기. (img & label 생성) 
#4. labels_txt 파일 전부 YoloStyle 로 변경
#4. 학습 시작.
'''
from dataset_utils import modify_data, augmentation, convert_yolostyle
import argparse
import os

def main(opt):
    train_images_path, val_images_path = ('./dataset/images/train/', './dataset/images/val/')
    train_json_label_path, val_json_label_path = ('./dataset/labels_json/train/', './dataset/labels_json/val/')
    train_txt_label_path, val_txt_label_path = ('./dataset/labels_txt/train/', './dataset/labels_txt/val/')
    train_yolo_label_path, val_yolo_label_path = ('./dataset/labels/train/', './dataset/labels/val/')
    
    no_match_train_img_path, no_match_val_img_path = ('./dataset/no_match/train/', './dataset/no_match/val/')
    train_augmentation_path, val_augmentation_path = ('./dataset/aug_data/train/', './dataset/aug_data/val/')
    
    print("start conver_json2txt")
    modify_data.convert_json2txt(train_json_label_path, train_txt_label_path) #1 #2
    modify_data.convert_json2txt(val_json_label_path, val_txt_label_path) #1 #2
    print("end conver_json2txt")
    print("start move_no_match_image")
    modify_data.move_no_match_image(train_txt_label_path, train_images_path, no_match_train_img_path)
    modify_data.move_no_match_image(val_txt_label_path, val_images_path, no_match_val_img_path)
    print("end move_no_match_image")
    print("start augmentation")
    augmentation.run(train_images_path, train_txt_label_path, train_augmentation_path) #3 
    augmentation.run(val_images_path, val_txt_label_path, val_augmentation_path) #3 
    print("end augmentation")
    print("start convert_yolostyle")
    print(1)
    convert_yolostyle.run(train_images_path, train_txt_label_path, train_yolo_label_path)
    print(2)
    convert_yolostyle.run(val_images_path, val_txt_label_path, val_yolo_label_path)
    print(3)
    convert_yolostyle.run(train_augmentation_path, train_augmentation_path, train_augmentation_path)
    print(4)
    convert_yolostyle.run(val_augmentation_path, val_augmentation_path, val_augmentation_path)
    print("end convert_yolostyle")

def parse_opt():
    parser = argparse.ArgumentParser(description='this is data augmentation program')
    parser.add_argument('--path', type=str, default='./')                  # yolo_path
    parser.add_argument('--img_path',  type=str, default='./dataset/images/')   # img train dir
    parser.add_argument('--label_path',  type=str, default='./dataset/labels/') # label train dir
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    PATH = opt.path
    os.chdir(PATH)
    print("PATH : {}".format(PATH))
    main(opt)
    ...

    # python data_preprocessing.py --path /home/woojin/_traffic_sig/yolov5/backup/ 
    '''
    dir_tree
    |d--  yolov5            
        |d-- dataset
            |d-- images      
                |d-- train
                |d-- val
            |d-- labels          # yolo_label
                |d-- train
                |d-- val
            |d-- labels_json     # raw_label
                |d-- train
                |d-- val
            |d-- labels_txt      # extract cls, bbox
                |d-- train
                |d-- val
            |d-- no_match        # no label img
            |d-- aug_data        # augmentation data

        |d-- pre_processing
            |f-- data_preprocessing.py *  # <- here
            |d-- dataset_util     
        
    '''