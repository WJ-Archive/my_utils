#현재 데이터셋 개수
#0 : Unknown - 약 37만장
#1 : red - 약 39만장
#2 : yellow(yellow + red_yellow + green_yellow + yellow_left_arrow) - 약 5만장
#3 : green : 약 50만장 
#4 : left (left(2000) + red_left_arrow(33000)) : 약 35000장
#5 : green_left : 약 6만장 
#6 : x_light : 약 5000장
#7 : v_light : 약 2만 6000장 

#> 데이터셋 불균형 이 심한 상태. 이중 x_light, v_light 의 개수는 늘려야 할 필요가 있고, Left 도 대부분 red_left 이고 left 는 부족함.
#그리고 Unknown 데이터셋 이 너무 애매함. 다 확인해보진 못했지만 신호를 판별 할 수 없던것들이 대부분 Unknown 이 들어가있었는데 데이터셋 개수도 많아서 추론할 때 이 클래스를 남발할 것 같음.

# import modules for image preprocessing
import cv2
import matplotlib.pyplot as plt
import os, sys
import argparse
import random
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import glob

# 데이터 샘플링 전에 어떤 클래스가 모자란지 확인 하기위해 Class Count
def count_cls(label_path:str) -> dict:
    class_dict = {}
    label_files = os.listdir(label_path)

    for file in label_files:
        with open(os.path.join(label_path, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls = line.split(' ')[0]

                # dict에 key 없으면 추가 초기화
                if cls not in class_dict:
                    class_dict[cls] = 0

                # 해당 key의 값을 1 증가
                class_dict[cls] += 1
    
    class_cnt = sorted(class_dict.items())
    print(class_cnt)

    return class_dict
    ...


def read_label_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    labels = []
    for line in lines:
        cls, x1, y1, x2, y2 = line.split()  # split the line into parts
        labels.append((int(cls), float(x1), float(y1), float(x2), float(y2)))
        
    return labels

def augmentation(img, bbs):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images

        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),

        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),

        iaa.Grayscale(alpha=(0.0, 1.0)),
    ], random_order=True) # apply augmenters in random order

    img_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)

    return img_aug, bbs_aug


def main(opt):
    #1. JSON 데이터셋 읽어와서 매칭되는 이미지와, 라벨링 파일 나누기
    #2. 읽어온 데이터셋에서 필요한 데이터셋만 뽑아내서 txt 파일로 변환 *cls x1 y1 x2 y2 형식으로 기록 (labels_txt 에 저장.)
    #3. 데이터 전처리 (PreProcessing). -> Sampling 후 YoloStyle 로 변경하여 labels directory에 저장. [here]
    #4. 데이터 학습 시작.

    img_path ='../dataset/labels_json/train/'
    label_path = '../dataset/labels_txt/train/'

    img_files = sorted(glob.glob(img_path+"*.jpg"))
    label_files = sorted(glob.glob(label_path+"*.txt"))    

    for i, (img_file, label_file) in enumerate(zip(img_files, label_files)):
        img = cv2.imread(img_file)
        height, width = img.shape[:2]  
        labels = read_label_file(label_file)


        bbs = ia.BoundingBoxesOnImage([
            # convert label values to pixel values
            ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=cls)
            for cls, x1, y1, x2, y2 in labels
        ], shape=img.shape)

        img_aug, bbs_aug = augmentation(img, bbs)

        # save augmented image and labels
        cv2.imwrite(f"./test_save/{i}.jpg", img_aug)
        with open(f"./test_save/{i}.txt", 'w') as f:
            for bb in bbs_aug.bounding_boxes:
                f.write(f"{bb.label} {bb.x1} {bb.y1} {bb.x2} {bb.y2}\n")











if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    ...





    
    
    


























"""
def vis(class_dict):
    # 데이터를 클래스와 카운트로 분리
    classes, counts = zip(*class_dict)

    # bar 그래프 생성
    plt.bar(classes, counts)

    # 그래프에 제목과 라벨 추가
    plt.title("Class Counts")
    plt.xlabel("Class")
    plt.ylabel("Count")

    # 그래프 보여주기
    plt.savefig("./class_counts.png")

"""