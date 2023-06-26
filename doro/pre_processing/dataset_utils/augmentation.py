from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import glob
import os
import numpy as np

def count_cls(label_path:str) -> dict:
    class_dict = {}
    label_files = os.listdir(label_path)

    for file in label_files:
        
        with open(os.path.join(label_path, file), 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                cls = line.split(' ')[0]

                if cls not in class_dict:
                    class_dict[cls] = 0
                class_dict[cls] += 1
            
    
    class_cnt = sorted(class_dict.items())
    #print(class_cnt)

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

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), 
        iaa.Flipud(0.2), 
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),

        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
            rotate=(-45, 45), 
            shear=(-16, 16), 
            order=[0, 1], 
            cval=(0, 255),
            mode=ia.ALL 
        )),

        #iaa.Grayscale(alpha=(0.0, 1.0)),
    ], random_order=True)

    img_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    bbs_aug = bbs_aug.clip_out_of_image()

    return img_aug, bbs_aug

def run(img_path:str, label_path:str, save_path:str):

    class_cnt = count_cls(label_path)
    cls_mean = np.mean(list(class_cnt.values()))
    #print(cls_mean)

    # count how many times we should augment each class
    augmentation_counts = {
        cls: max(0, int(round(cls_mean / count))-1) 
        for cls, count in class_cnt.items()
    }
    

    label_files = sorted(glob.glob(label_path+"*.txt"))
    #print(label_files)
    #label 이 존재하는 파일만 가져오기위해 img_files glob 이 아니라 label_files 를 대체.
    
    img_files = [os.path.join(img_path, os.path.basename(f).replace('.txt', '.jpg')) for f in label_files]
    #img_files = sorted(glob.glob(img_path+"*.jpg"))
    #print(img_files)
    for i, (img_file, label_file) in enumerate(zip(img_files, label_files)):
        try:
            img = cv2.imread(img_file)
            labels = read_label_file(label_file)

            #Filtering aug_label -> Augmetation 이 필요한 class의 Label 만 남겨놓은 채 나머지 Label은 삭제.
            aug_labels = [label for label in labels if str(label[0]) in augmentation_counts and augmentation_counts[str(label[0])] != 0]

            #Filtering 된 라벨만 가지고 Augmentation 수행. 그중 가장 큰 Value 값을 가진 cls 를 기준으로 반복문 수행. 가장 cls 개수가 적은 6이 포함되있을 경우 35 번 반복되기때문에 어느정도 데이터의 균형을 맞출 수 있지않을까..
            if len(aug_labels) != 0:
                aug_cls = [aug_label[0] for aug_label in aug_labels]
                aug_cnt = max(augmentation_counts.get(str(cls), 0) for cls in aug_cls)
                #print("now cls , cnt : ",aug_cls, aug_cnt)

                for _ in range(aug_cnt):
                    
                    bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=cls) for cls, x1, y1, x2, y2, in aug_labels], shape=img.shape)

                    # Augment
                    img_aug, bbs_aug = augmentation(img, bbs)
                    # Save augmented image and labels
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    output_index = len(glob.glob(f"{save_path}/*.jpg"))
                    cv2.imwrite(f"{save_path}/{output_index}.jpg", img_aug)
                    with open(f"{save_path}/{output_index}.txt", 'w') as f:
                        for bb in bbs_aug.bounding_boxes:
                            f.write(f"{bb.label} {bb.x1} {bb.y1} {bb.x2} {bb.y2}\n")
                    #print("save : {}\n{}\n".format(output_index, bbs_aug.bounding_boxes))

        except Exception as e:
            print("????")
            print(e)
