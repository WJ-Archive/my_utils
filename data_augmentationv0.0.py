# auto Data Augmentation Tool v0.0
# 2. Scale
# 3. flip
# 4. Shift or Translate
# 5. Rotate
# 6. add a noise

# import modules for image preprocessing
import cv2
import matplotlib.pyplot as plt
import os, sys
import argparse
import random
import numpy as np

#
class Augmentation:
    def __init__(self, source_path:str, save_path:str, selected_aug:list, repeat : int, aug_mix:bool, set_rate:bool):
 
        self.source_path = source_path       
        self.save_path = save_path  
        self.selected_aug = selected_aug         
        self.repeat = repeat
        self.aug_mix = aug_mix  
        self.set_rate = set_rate  
        
        #augmentation rate
        #TODO : 적당한 선에서 random 값 넣기
        
        # self.scale_rate = (random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)) # fx : 0~1, fy : 0~1 (%)
        self.scale_rate = (0.3, 0.7) # fx : 0~1, fy : 0~1 (%)
        self.flip_type = 1 # -1, 0, 1
        self.translate_loc = (50, 50) #x, y 
        self.rotate_angle = 10 #angle
        self.noise = 1 # add noise
        self.rand_crop_size = 640
        if set_rate: self.setting_rate()

        #'scale', 'flip', 'translate', 'rotate', 'add_noise', 'rand_crop', 
        self.augmente_dict = {
            'scale': self.cv_scale, 'flip':self.cv_flip, 'translate':self.cv_translate, 
            'rotate':self.cv_rotate, 'add_noise':self.cv_add_noise, 'rand_crop':self.cv_rand_crop
            }

    def setting_rate(self):
        print("set_rate True/./.")
        ...

    def data_augmentation(self):
        files = os.listdir(self.source_path)
        print("file total nums {}".format(len(files)))
        try:
            for i in range(self.repeat):

                for file_name in files:
                    # print("filename :",file_name)
                    file_path = os.path.join(self.source_path, file_name)
                    img = cv2.imread(file_path)
                    
                    # execute select function 
                    for j, aug in enumerate(self.selected_aug):
                        print(j, aug)
                        cv_img, cv_type = self.augmente_dict[aug](img) #img: numpy , rate: float  -> cv_img: numpy (convert image)

                        if self.aug_mix: #aug mix == True
                            img = cv_img 
                            if j != len(self.selected_aug)-1:
                                continue
                            else:
                                self.save(file_name, img, '_'+str(i)+'_'+'_'.join(self.selected_aug))
                        else:
                            self.save(file_name, cv_img, '_'+str(i)+cv_type)
                
        except Exception as e:
            print(e)
            return "Failed"    

    def cv_scale(self, img) -> tuple:# 스케일링의 경우 얼굴인식 처럼 자세히 봐야하는 경우 특징이 변형 될 수 있음. 

        # (f_x, f_y) = self.scale_rate
        f_x = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        f_y = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        try:
            cv_type = "_scale_"
            scaled_img = cv2.resize(img, None, fx=f_x, fy=f_y, interpolation = cv2.INTER_CUBIC)
            return scaled_img, cv_type
        
        except Exception as e:
            print(e)
            return "Scale Failed"
            
    def cv_flip(self, img) -> tuple:

        self.flip_type = random.choice([-1,0,1])

        try:
            if self.flip_type == 0:
                cv_type = "_flip_horizen_"
            elif self.flip_type == 1:
                cv_type = "_flip_vertical_"
            elif self.flip_type == -1:
                cv_type = "_flip_bothFlip_"
            else:
                print("flip type must be -1, 0, 1")
                return 0
            
            flip_img = cv2.flip(img, self.flip_type)
            return flip_img, cv_type

        except Exception as e:
            print(e)
            return "Flip Failed"
        ...

    def cv_translate(self, img) -> tuple:
        # (x, y) = self.translate_loc
        x = random.randrange(-100, 100)
        y = random.randrange(-100, 100)
        try:
            cv_type = "_translate_"
            (h, w) = img.shape[:2]
            transform_matrix = np.float32([[1,0,x],[0,1,y]])    #변환 행렬 생성 
            translate_img = cv2.warpAffine(img, transform_matrix, (w, h))   #shift img 
            ...
            return translate_img, cv_type
        except Exception as e:
            print(e)
            return "Translate Failed"
        
    def cv_rotate(self, img) -> tuple:
        #angle = self.rotate_angle
        angle = random.choice([-15,-10,10,15])
        try:
            cv_type = "_rotate_"+str(angle)+"_"
            (h, w) = img.shape[:2]
            (center_x, center_y) = (w//2, h//2)                                               
            rotation_matrix = cv2.getRotationMatrix2D((center_x ,center_y), angle, 1)     #center, angel, scale
            rotate_img = cv2.warpAffine(img, rotation_matrix, (w, h))
            return rotate_img, cv_type
        
        except Exception as e:
            print(e)
            return "rotate Failed"

    def cv_add_noise(self, img) -> tuple:
        # 일단 가장 널리 쓰이는 가우시안 노이즈 추가. 평균이 mean 이고 표준편차가 std 인 노이즈 이미지 생성후 add로 합침.
        mean = 0 
        std = 1
        #noise_type = 'gauss'
        try:
            cv_type = "_noisy_"
            # Generate Gaussian noise
            gauss = np.random.normal(mean, std, img.shape).astype('uint8')

            # Add the Gaussian noise to the image
            noisy = cv2.add(img, gauss)
            return noisy,cv_type
        except Exception as e:
            print(e)
            return "add noise Failed"
        ...

    def cv_rand_crop(self, img) -> tuple:
        '''
        rand crop
        '''
        #save(save_path, file_name, cv_img)

    def save(self, file_name, cv_img, cv_type) -> None:
        '''
        save convert image
        if cv_img list 
        '''
        if os.path.isdir(self.save_path) != True:
            os.mkdir(self.save_path)
        
        saved_name = os.path.join(self.save_path,"{}{}.{}".format(file_name.split('.')[0], cv_type, 'jpg'))
        print(f"save img : {saved_name}")
        cv2.imwrite(saved_name, cv_img)

def main(opt):
    SOURCE_PATH = opt.source_path
    SAVE_PATH = opt.save_path
    selected_aug = vars(opt)['augmentation']

    if(len(selected_aug) == 0):
        
        print("\n************************************************************************")
        print("## select augmentation option ##")
        print("option :: [ resize, scale, flip, translate, rotate, add_noise, rand_crop ]")
        print("if want multiple augmentations together, use --aug_mix args\n\n")
        print("rate default is random but using --rate option can setting rate")
        print("************************************************************************\n")
        print("example) >> python data_augmentation.py --augmentation scale, flip\n")
        
        ...

    aug = Augmentation(SOURCE_PATH, SAVE_PATH, selected_aug, opt.repeat, opt.aug_mix, opt.set_rate)
    aug.data_augmentation()
    
def parse_opt():
    parser = argparse.ArgumentParser(description='this is data augmentation program')
    parser.add_argument('--source_path', type=str, default='./_src_imgs/')
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
