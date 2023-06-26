import os
import json
import shutil

def delete_empty_files(path='./'):
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.getsize(filepath) == 0:
            os.remove(filepath)

#1
def write_cls_bbox(cls:str, y_xywh:list, fp, save_path='./'): 
    
    yx, yy, yw, yh = map(str,y_xywh)
    fp.write(' '.join([cls, yx, yy, yw, yh]))
    fp.write('\n')


def generate_class(obj:dict) -> tuple: 
    try:
        #Class Mapping Dictionary
        t_class_map = {'unknown':'0',                                              
                    'red':'1',
                    'yellow':'2','red_yellow':'2','green_yellow':'2','yellow_left_arrow':'2',
                    'green':'3',
                    'left_arrow':'4','red_left_arrow':'4',
                    'green_left_arrow':'5',
                    'x_light':'6','others_arrow':'7'}
        t_class = None

        for attr in obj['attribute']:
            traffic_signal = [key for key, value in attr.items() if value == "on"] #str
            bbox = obj['box'] #xyxy
            if(len(traffic_signal) == 1):   
                t_class = t_class_map[''.join(traffic_signal)]
                if t_class != None:    
                    return t_class, bbox
                       
            elif(len(traffic_signal) > 1):  
                mul_class = '_'.join(traffic_signal)                    
                t_class = t_class_map[mul_class] if mul_class in t_class_map else None #t_class_map에 있을경우만 write. 아니면 skip 
                if t_class != None:                                         
                    return t_class, bbox
                else:                                                       
                    continue
            else:                           
                t_class = t_class_map['unknown'] 
                if t_class != None:                    
                    return t_class, bbox

    except Exception as e:
        print(e)
        ...

def convert_json2txt(source_label_path, txt_label_path):

    files = os.listdir(source_label_path)
    try:
        for json_file_name in files:
            
            file_path = os.path.join(source_label_path, json_file_name)
            with open(file_path, "r") as str_json:
                label_json = json.load(str_json)
            image_inf = label_json['image']
            write_txt_list = []
            for obj in label_json['annotation']:
                if obj['class'] == 'traffic_light': 
                    write_txt_list.append(generate_class(obj))
                else:
                    continue
            if len(write_txt_list) != 0:
                with open(txt_label_path+json_file_name[:-4]+"txt", "w") as fp:
                    for l in write_txt_list:
                        if l is not None: 
                            #print(l[0], l[1], type(l[0]),type(l[1]))
                            write_cls_bbox(l[0], l[1], fp)
                
        delete_empty_files(txt_label_path)
        
    except Exception as e:
        print(e)

def move_no_match_image(label_folder:str, img_folder:str, no_match_folder:str):
    try:
        label_files = os.listdir(label_folder)  #[1.txt,2.txt,3.txt,4.txt,5.txt]
        img_files = os.listdir(img_folder)      #[1.jpg,2.jpg,3.jpg,4.jpg,5.jpg,6.jpg,7.jpg]

        label_basenames = set(os.path.splitext(name)[0] for name in label_files if name.endswith(".txt"))
        img_basenames = set(os.path.splitext(name)[0] for name in img_files if name.endswith(".jpg"))
        print(label_basenames)
        print(img_basenames)
        if len(label_basenames) < len(img_basenames):
            no_match_imgs = img_basenames - label_basenames #[6.jpg,7.jpg]
        else:
            no_match_imgs = label_basenames - img_basenames #[6.jpg,7.jpg]
        

        for img in no_match_imgs:
            src_path = os.path.join(img_folder, img + '.jpg')
            dest_path = os.path.join(no_match_folder, img + '.jpg')
            shutil.move(src_path, dest_path)
            
    except Exception as e:
        print(e)

