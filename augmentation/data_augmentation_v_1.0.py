# import modules for image preprocessing
import cv2
import matplotlib.pyplot as plt
import os, sys
import argparse
import random
import numpy as np

#plan
#라벨링 데이터도 같이 Augmentation 하기위해선...
# 00001.jpg mapping txt 파일 예시
# 00001.txt 
# 0 0.123023 1.23232 3.123123 4.123123
# 1 1.123123 1.123123 1.4242 3.123123
#1. Labeling Directory 에서 데이터 읽은후, 같은 이름의 이미지 파일 리스트 에 Dictinary로 mapping 
#ex) {"00001.jpg":{{'0':['0.123023','1.23232','3.123123','4.123123']},{'1':['1.123123','1.123123','1.4242','3.123123']}...}

#2. 이미지 Scaling 후 Value 안의 bbox 값도 같은 비율로 convert. 
#ex) flip 될 경우 bbox 값도 flip, rotate 했을때는 회전해서 증가한 xyxy 만큼 bbox의 위치를 늘리거나 줄여야함ㄷㄷ. Noise 의 경우에는 변동 없음,

# Basic image manipulation 만 구현
# 그중 Geometric Transformation(flip, rotate, contrast)와 Color Space transformations(gray,hsv) 만 구현
# Mixing images 같은 기법은 Yolov5에서 Training 할때 어느정도 Augmentation 후 학습하기때문에 한다면 후순위로..

class Augmentation_Setting:
    def __init__(self, img_path:str, label_path:str, save_path:str, set_rate:bool):
 
        self.img_path = img_path       
        self.save_path = save_path  
        self.label_path = label_path
        self.set_rate = set_rate

        self.files = self.img_label_mapping() if label_path != '' else os.listdir(self.img_path) #-> label_path!='' ? dict : list
        if self.set_rate: self.setting_rate()

    def setting_rate(self):
        #TODO Scale 설정 하는 함수. 나중에 사용자 설정에 따라 변경 할 수 있는 UI 추가 (지금은 default 값으로 설정)
        print("default set..")
        # Geometric_Transformation default rate
        self.resize_fx, self.resize_fy = (0.3, 0.7) # fx : 0~1, fy : 0~1 (%)
        self.flip_type = 1 # -1, 0, 1
        self.translate_x, self.translate_y = (50, 50) #x, y 
        self.rotate_angle = 10 #angle
        self.mean, self.noise, self.noise_type = (0, 1, 'gauss') # add noise
        
    def rate_shuffle(self, aug_type):
        print("random shuffle..")
        # Geometric_Transformation default rate
        if aug_type == '_resize_':
            self.resize_fx = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            self.resize_fy = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        elif aug_type == '_flip_':    
            self.flip_type = random.choice([-1, 0, 1])
        elif aug_type == '_translate_':
            self.translate_x = random.randrange(-100, 100)
            self.translate_y = random.randrange(-100, 100)
        elif aug_type == '_rotate_':
            self.rotate_angle = random.choice([-15,-10,10,15]) #angle
        elif aug_type == '_noise_':
            self.mean, self.std, self.noise_type = (0, 1, random.choice(['gauss']))
        else:
            print("no rat")
            ...
            # Color Space Transformation Setting
        ...

    def check_class_num():#TODO : 클래스 개수 체크하고 데이터 불균형 일 경우 알아서 샘플링 하도록 변경
        ...

    def read_label(self, label_path:str)->dict:
        labels = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                split_line = line.strip().split()
                labels[split_line[0]] = split_line[1:]
        return labels

    def img_label_mapping(self)->dict:
        img_dir_path = self.img_path
        label_dir_path = self.label_path
        img_label_map = {}

        for img_file in os.listdir(img_dir_path):
            if img_file.endswith('.jpg'):
                
                img_path = os.path.join(img_dir_path, img_file)                             #현재 이미지파일의 path
                label_path = os.path.join(label_dir_path, img_file.replace('.jpg','.txt'))  #현재 이미지파일의 매칭 되는 txt 파일 path

                if os.path.isfile(label_path):                                              #매칭되는 파일이 있을경우. mapping dictionary 생성
                    labels = self.read_label(label_path)                                         
                    img_label_map[img_path] = labels                                        
                    '''
                    ex)
                    {'./test_case/s01503172.jpg': {'0': ['0.057813', '0.281944', '0.009375', '0.022222'], '1': ['0.474609', '0.128472', '0.049219', '0.043056']}, 
                    './test_case/s01503173.jpg': {'0': ['0.470313', '0.222917', '0.023438', '0.009722'], '3': ['0.534766', '0.213194', '0.022656', '0.006944']}, 
                    './test_case/s01503174.jpg': {'1': ['0.552734', '0.333333', '0.017969', '0.016667']}, ...}
                    '''

        return img_label_map

#Basic_Image_Manipulation
class Geometric_Transformation(Augmentation_Setting):

    def __init__(self, parent=None):
        super().__init__(parent.img_path, parent.label_path, parent.save_path, parent.set_rate)

    def cv_resize(self, img:np.array, cls_label:dict='') -> tuple:

        try:
            cv_type = "_resize_"
            if ~self.set_rate:
                self.rate_shuffle(cv_type) #random setting
                f_x, f_y = (self.resize_fx, self.resize_fy)
            else:
                f_x, f_y = (self.resize_fx, self.resize_fy)

            scaled_img = cv2.resize(img, None, fx=f_x, fy=f_y, interpolation = cv2.INTER_CUBIC)
            
            # resize 된 만큼 라벨링 변경
            if cls_label is not None:
                # 어차피 다 640으로 resize 해서 별로 상관없었나..
                #scaled_label = {k: [float(v[0])*f_x, float(v[1])*f_y, float(v[2])*f_x, float(v[3])*f_y] for k, v in cls_label.items()}
                scaled_label = {k: [float(v[0]), float(v[1]), float(v[2]), float(v[3])] for k, v in cls_label.items()}
                print(scaled_label)
            else:
                scaled_label = cls_label
                
            return scaled_img, scaled_label, cv_type
        
        except Exception as e:
            print("error in {}".format(e))
            return "Scale Failed"
            
    def cv_flip(self, img:np.array, cls_label:dict='') -> tuple:
        try:
            cv_type = '_flip_'
            
            if ~self.set_rate:
                self.rate_shuffle(cv_type) #random shuffle
                flip_type = self.flip_type
            else:
                flip_type = self.flip_type

            flip_img = cv2.flip(img, flip_type)
            
            if cls_label is not None:
                if flip_type == 1: # horizontal flip
                    flip_label = {k: [1 - float(v[0]), float(v[1]), float(v[2]), float(v[3])] for k, v in cls_label.items()}
                elif flip_type == 0: # vertical flip
                    flip_label = {k: [float(v[0]), 1 - float(v[1]), float(v[2]), float(v[3])] for k, v in cls_label.items()}
                elif flip_type == -1: # flip both axes
                    flip_label = {k: [1 - float(v[0]), 1 - float(v[1]), float(v[2]), float(v[3])] for k, v in cls_label.items()}
            else:
                flip_label = cls_label

            return flip_img, flip_label, cv_type

        except Exception as e:
            print("error in {}".format(e))
            return "Flip Failed"
        ...

    def cv_translate(self, img:np.array, cls_label:dict='') -> tuple:
        #TODO : 이미지 이동시키다 바운딩 박스 밖으로 벗어나는 문제 발생. 체크후 바운딩 박스 안 벗어나게 하는 코드 작성 필요...
        try:
            yolo_label = True
            cv_type = "_translate_"
            if ~self.set_rate:
                self.rate_shuffle(cv_type) #random shuffle
                x, y = self.translate_x, self.translate_y
            else:
                x, y = self.translate_x, self.translate_y

            (h, w) = img.shape[:2]
            transform_matrix = np.float32([[1,0,x],[0,1,y]])    #변환 행렬 생성 
            translate_img = cv2.warpAffine(img, transform_matrix, (w, h))   #shift img 
            if cls_label is not None:
                
                if yolo_label == True:
                    # YoloStyle label은 정규화가 되어있어서 pixel 단위 만큼 이동시킨거라 Pixel 단위만큼 이동시킨 이 코드에서는 한번 정규화 를 거쳐야함.
                    lb_x_translated = float(x) / w
                    lb_y_translated = float(y) / h
                else: 
                    lb_x_translated = float(x)
                    lb_y_translated = float(y)

                # Translate the labels
                translate_label = {k: [float(v[0]) + lb_x_translated, float(v[1]) + lb_y_translated, float(v[2]), float(v[3])] for k, v in cls_label.items()}
            else:
                translate_label = cls_label
            
            return translate_img, translate_label, cv_type
        
        except Exception as e:
            print("error in {}".format(e))
            return "Translate Failed"
        
    def cv_rotate(self, img:np.array, cls_label:dict='') -> tuple:
        try:
            # cv_type = "_rotate_"+str(angle)+"_"
            cv_type = "_rotate_"
            if ~self.set_rate:
                self.rate_shuffle(cv_type) #random shuffle
                angle = self.rotate_angle
            else:
                angle = self.rotate_angle

            (h, w) = img.shape[:2]
            (center_x, center_y) = (w//2, h//2)                                               
            rotation_matrix = cv2.getRotationMatrix2D((center_x ,center_y), angle, 1)     #center, angel, scale
            rotate_img = cv2.warpAffine(img, rotation_matrix, (w, h))
            
            rotate_label = ...

            return rotate_img, rotate_label, cv_type
        
        except Exception as e:
            print("error in {}".format(e))
            return "rotate Failed"

    def cv_add_noise(self, img:np.array, cls_label:dict='') -> tuple:
        # 일단 가장 널리 쓰이는 가우시안 노이즈 추가. 평균이 mean 이고 표준편차가 std 인 노이즈 이미지 생성후 add로 합침.
        try:
            cv_type = "_noise_"
            if ~self.set_rate:
                self.rate_shuffle(cv_type) #random shuffle
                mean, std, noise_type = self.rotate_angle
            else:
                mean, std, noise_type = self.rotate_angle

            # Generate Gaussian noise
            gauss = np.random.normal(mean, std, img.shape).astype('uint8')
            # Add the Gaussian noise to the image
            noisy = cv2.add(img, gauss)
            
            return noisy, cls_label, cv_type
        
        except Exception as e:
            print("error in {}".format(e))
            return "add noise Failed"
        ...

class Color_Space_transformation(Augmentation_Setting):
    def __init__(self, parent=None):
        super().__init__(parent.img_path, parent.label_path, parent.save_path, parent.set_rate)

    def cv_gray_scale(self, img:np.array, cls_label:dict='') -> tuple:
        
        ...

class Mixing_images:
    ...

def pre_view():
    ...

def save_cvlabel(save_path:str, file_name:str, cv_label:dict, cv_type:str) -> None:
    # 디렉토리 없으면 생성 (하위디렉토리까지)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    saved_name = os.path.join(save_path,"{}{}.{}".format(file_name.split('.')[0], cv_type, 'txt'))
    
    # Write labels to text file
    with open(saved_name, 'w') as f:
        for k, v in cv_label.items():
            f.write('{} {}\n'.format(k, ' '.join(map(str, v))))
        print(f"save txt : {saved_name}")
    ...


def save_cvimg(save_path:str, file_name:str, cv_img:np.array, cv_type:str) -> None:
        '''
        save convert image
        if cv_img list 
        '''
        # 디렉토리 없으면 생성 (하위디렉토리까지)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        saved_name = os.path.join(save_path,"{}{}.{}".format(file_name.split('.')[0], cv_type, 'jpg'))
        cv2.imwrite(saved_name, cv_img)
        print(f"save img : {saved_name}")

def main(opt):

    IMG_PATH = opt.img_path
    SAVE_PATH = opt.save_path
    LABEL_PATH = opt.label_path
    selected_aug = vars(opt)['augmentation']
    print("selecterd_aug{}".format(selected_aug))

    if(len(selected_aug) == 0):
        
        print("\n************************************************************************")
        print("## select augmentation option ##")
        print("option :: [ resize, scale, flip, translate, rotate, add_noise, rand_crop ]")
        print("if want multiple augmentations together, use --aug_mix args\n\n")
        print("rate default is random but using --rate option can setting rate")
        print("************************************************************************\n")
        print("example) >> python data_augmentation.py --augmentation scale, flip\n")
        return 0 

    aug_config = Augmentation_Setting(IMG_PATH, LABEL_PATH, SAVE_PATH, opt.set_rate)

    augmente_dict = {
        #Geometric_Transformation
        'resize': Geometric_Transformation(aug_config).cv_resize, 'flip':Geometric_Transformation(aug_config).cv_flip, 
        'translate':Geometric_Transformation(aug_config).cv_translate, 'rotate':Geometric_Transformation(aug_config).cv_rotate, 
        'add_noise':Geometric_Transformation(aug_config).cv_add_noise

        #Color_Space_transformation
        #'gray_scale': Color_Space_transformation(aug_config).cv_gray
    }

    if isinstance(aug_config.files, dict):           #self.files 가 dictionary type 일 경우, 즉 label_path 가 들어왔을 경우.
        try:
            for i in range(opt.repeat):
                for img_path, cls_lable in aug_config.files.items():

                    file_name = img_path.split('\\')[-1]
                    img = cv2.imread(img_path)
                    for j, aug in enumerate(selected_aug):
                        cv_img, cv_cls_label, cv_type = augmente_dict[aug](img, cls_lable)
                        if opt.aug_mix:
                            img = cv_img
                            if j != len(selected_aug)-1:
                                continue
                            else:
                                save_cvimg(SAVE_PATH, file_name, img, '_'+str(i)+'_'+'_'.join(selected_aug))
                                #save_cvlabel(SAVE_PATH, )
                        else:
                            
                            save_cvimg(SAVE_PATH, file_name, cv_img, '_'+str(i)+cv_type)
                            save_cvlabel(SAVE_PATH, file_name, cv_cls_label, '_'+str(i)+cv_type)

        except Exception as e:
            print("error indasd {}".format(e))
        ...
    else:                                           # self.files가 list type 일 경우, 즉 label path 없이 이미지 경로만 있는 경우
        try:
            files = os.listdir(IMG_PATH)
            for i in range(opt.repeat):
                for file_name in files:
                    if file_name.endswith('.jpg'):
                        print("filename :",file_name)
                        file_path = os.path.join(IMG_PATH, file_name)
                        img = cv2.imread(file_path)
                        
                        # execute select function 
                        for j, aug in enumerate(selected_aug):
                            cv_img, _, cv_type = augmente_dict[aug](img)
                            if opt.aug_mix:
                                img = cv_img
                                if j != len(selected_aug)-1:
                                    continue
                                else:
                                    save(SAVE_PATH, file_name, img, '_'+str(i)+'_'+'_'.join(selected_aug))
                            else:
                                save(SAVE_PATH, file_name, cv_img, '_'+str(i)+cv_type)
                
        except Exception as e:
            print("error inasdasd {}".format(e))
            return "Failed"    

def parse_opt():
    parser = argparse.ArgumentParser(description='this is data augmentation program')
    parser.add_argument('--img_path', type=str, default='./')
    parser.add_argument('--label_path',  type=str, default='', help='If there is a labeled directory, write down the path and it will be matched with the filename, and augmented with the bbox.')
    parser.add_argument('--save_path', type=str, default='./_cv_imgs/')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--augmentation', nargs = '+', type=str, default='', help='resize, scale, flip, translate, rotate, add_noise, rand_crop')
    parser.add_argument('--aug_mix', action='store_true', help='if want multiple augmentations together, use this option')
    parser.add_argument('--set_rate', action='store_true', help='for setting rate, default True')
    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    ...
