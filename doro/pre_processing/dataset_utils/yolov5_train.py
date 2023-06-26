from glob import glob
from sklearn.model_selection import train_test_split
import yaml
import os 

def divide_dataset(img_list:list):
    '''
    만약 데이터셋이 Train 과 Valid로 나눠지지 않았다면 이 함수로 나누고 시작
    '''
    train_img_list, valid_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)
    return train_img_list, valid_img_list
    ...

def write_img_path(train_img_list:list, valid_img_list:list):
    '''
    Training 이미지 경로와 Validation 이미지 경로를 txt 파일로 저장
    '''
    with open ('./dataset/'+'train.txt', 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')

    with open ('./dataset/'+'val.txt', 'w') as f:
        f.write('\n'.join(valid_img_list) + '\n')
    ...

def mk_yaml(yaml_path:str, train_path:str, val_path:str, nc:int, names:list):
    '''
    yaml 파일 존재하지 않는 경우 yaml 파일 생성.
    '''
    
    with open(yaml_path,'w') as yf:
        yf.write("train: {}\n".format(train_path))
        yf.write("val: {}\n".format(val_path))
        yf.write("nc: {}\n".format(str(nc)))
        yf.write("names: {}\n".format(str(names))) #여기선 [] 도 들어가야하니 ','.join이 아닌 str()을 씀
        yf.write("\n")

def mod_yaml(yaml_path:str, train_path:str, val_path:str, nc:int, names:list):
    '''
    yaml 파일있는 경우 수정
    '''

    with open(yaml_path,'r') as yf:
        data = yaml.full_load(yf)

    print(data)
    data['names'] = names # list 는 yaml에서 알아서 변환되기 때문에 str 필요 없음
    data['nc'] = str(nc)
    data['val'] = val_path
    data['train'] = train_path
    
    with open(yaml_path, 'w') as yf:
        yaml.dump(data, yf)


'''
# Yolov5 의 기본 학습 경로
|-- yolov5
    |-- dataset
        |--images
            |-- train
            |-- val
        |--labels
            |-- train
            |-- val

# yaml 파일에서 datasets 의 경로를 바꿔줄순 있음.
    ex) data.yaml
    train: ../coco128/images/train2023/  # 128 images
    val: ../coco128/images/train2023/  # 128 images

# 단 yaml 파일에서 train과 val 경로를 바꾸는건 불가능 (Yolov5의 데이터 로딩 메커니즘 문제)
# 굳이 바꾼다면 Dataloaders.py 에서 변경이 가능하긴 한데 복잡하니 그냥 위의 경로를 따르는 게좋다.
    ex)
    ../coco128/images/train2017/000001.jpg 이미지에 대한 label 파일은 ../coco128/labels/train2017/000001.txt에 위치
'''

'''
|-- yolov5
    |-- dataset_util
        |-- *(this_file)
    |-- dataset
        |--images
            |-- train
            |-- val
        |--labels
            |-- train
            |-- val

'''

def count_cls(label_path:str) -> dict:
    class_dict = {}
    label_files = os.listdir(label_path)
    print(label_path)

    for file in label_files:
        #print(file)
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

def check_no_match_image(label_folder:str, img_folder:str):
    try:
        label_files = os.listdir(label_folder)  #[1.txt,2.txt,3.txt,4.txt,5.txt]
        img_files = os.listdir(img_folder)      #[1.jpg,2.jpg,3.jpg,4.jpg,5.jpg,6.jpg,7.jpg]

        label_basenames = set(os.path.splitext(name)[0] for name in label_files if name.endswith(".txt"))
        img_basenames = set(os.path.splitext(name)[0] for name in img_files if name.endswith(".jpg"))

        no_match_imgs = img_basenames - label_basenames #[6.jpg,7.jpg]

        print("no matching img : ",no_match_imgs)
            
    except Exception as e:
        print(e)

def main(YOLO_PATH='./'):

    os.chdir(YOLO_PATH) #경로 yolo_path 로 변경

    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    TRAIN_IMG_PATH = './backup/dataset/images/train/'
    VALID_IMG_PATH = './backup/dataset/images/val/'
    TRAIN_TXT_PATH = './backup/dataset/labels/train/'
    VALID_TXT_PATH = './backup/dataset/labels/val/'
    train_img_list = glob(TRAIN_IMG_PATH+'*.jpg')
    valid_img_list = glob(VALID_IMG_PATH+'*.jpg')
    #Number of Class
    NC = 8

    #CLS NAMES   0       1        2         3        4        5           6            7   
    NAMES = ['unknown', 'red', 'yellow', 'green', 'left', 'green_left', 'x_light', 'v_light']

    #yaml 생성/수정
    if os.path.exists('./backup/dataset/data.yaml'): #bool
        #data.yaml 존재하는지 확인후 있으면 수정
        mod_yaml('./backup/dataset/data.yaml', TRAIN_IMG_PATH, VALID_IMG_PATH, NC, NAMES)
        print('modify yaml')

    else:
        #없으면 data.yaml 생성
        mk_yaml('./backup/dataset/data.yaml', TRAIN_IMG_PATH, VALID_IMG_PATH, NC, NAMES)
        print('generate yaml')

    #matching 안된 파일은 전부 no_match 폴더로 옮김
    check_no_match_image(TRAIN_IMG_PATH, TRAIN_TXT_PATH)
    check_no_match_image(VALID_IMG_PATH, VALID_TXT_PATH)
    #데이터 불균형이 있는지 확인.
    count_cls(TRAIN_TXT_PATH)
    count_cls(VALID_TXT_PATH)

    with open("./backup/dataset/train.txt","w") as t:
        for file in train_img_list:
            t.write("{}\n".format(file))

    with open("./backup/dataset/valid.txt","w") as t:
        for file in valid_img_list:
            t.write("{}\n".format(file))




    """
    #학습할 데이터 train.txt 에 작성 (원래는 이미지 폴더 glob 하지만 여기선 labeling 안된파일이 다수 존재하여 label txt 파일 생성된 이미지만 넣기 위해 label 파일 glob 후 확장자 변경해서 작성)
    #file name ex) 13646365.txt
    IMG_PATH = './dataset/images/'
    train_list = glob(TRAIN_TXT+'*.txt')
    valid_list = glob(VALID_TXT+'*.txt')

    with open("./dataset/train.txt","w") as t:
        for file in train_list:
            t.write("{}\n".format(IMG_PATH+file[17:-4]+'.jpg')) 
        ...

    with open("./dataset/valid.txt","w") as t:
        for file in valid_list:
            t.write("{}\n".format(IMG_PATH+file[17:-4]+'.jpg')) 
    
    """

if __name__ == "__main__":
    
    YOLO_PATH = '/home/woojin/_traffic_sig/yolov5/'
    print("path",YOLO_PATH)
    main(YOLO_PATH)


#python train.py --img 640 --batch 16 --epochs 100 --data ./dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolov5s_traffic --device 0,1 --project traffic_aug_1
#python train.py --img 640 --batch 16 --epochs 100 --data ./backup/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolov5s_traffic --device 6,7 --project traffic_no_mod_1