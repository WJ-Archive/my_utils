import json 
import os
import shutil

def convert_yolo_style(xyxy:list, img_size:list) -> list:
    '''
    Labeling 된 BBOX의 x1, y1, x2, y2 좌표를 YoloStyle 로 변환 하는 코드
    xywh 라면 밑에 w와 h 계산할 필요 없이 바로 넣는다.
    '''
    
    (x1, y1, x2, y2) = xyxy
    img_w, img_h = img_size
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


def write_cls_bbox(cls:str, y_xywh:list, fp, save_path='./'): 
    '''
    Yolo Label txt 파일로 write 하는 함수

    밖에서 파일포인터 열고 포인터를 argument로 받아 여기서 작업 후 
    끝나면 밖에서 다시 닫는 식으로 사용해야함. (함수 들어올때마다 파일을 열면 속도저하 문제 발생)
    데이터셋마다 json 형식이 달라서 일단 이 방식이 최선인듯 
    '''
    yx, yy, yw, yh = map(str,y_xywh)
    fp.write(' '.join([cls, yx, yy, yw, yh]))
    fp.write('\n')
    #print(" {} : write Success".format(str(fp)))


def generate_label_file(obj:dict, json_file_name:str, image_inf:dict, destination_path:str='./') -> str: 

    #Class Mapping Dictionary
    t_class_map = {'unknown':'0',                                              
                   'red':'1',
                   'yellow':'2','red_yellow':'2','green_yellow':'2','yellow_left_arrow':'2',
                   'green':'3',
                   'left_arrow':'4','red_left_arrow':'4',
                   'green_left_arrow':'5',
                   'x_light':'6','others_arrow':'7'}
    t_class = None

    try:
        #txt file open
        with open(destination_path+json_file_name[:-4]+"txt", "a") as fp:

            for attr in obj['attribute']:
                
                traffic_signal = [key for key, value in attr.items() if value == "on"] #str
                bbox = obj['box'] #xyxy

                if(len(traffic_signal) == 1):   #traffic_signal이 하나일때
                    t_class = t_class_map[''.join(traffic_signal)]
                    if t_class != None:     
                        #print("cls : {} box :{}".format(t_class, ybox))               
                        write_cls_bbox(t_class, bbox, fp)              

                elif(len(traffic_signal) > 1):  
                    #만약 attribute 가 동시에 2개 이상 있다면 2가지 값을 합쳐서 클래스를 만듬. (ex. red, left_arrow => red_left_arrow : 0)
                    mul_class = '_'.join(traffic_signal)                    

                    # 2개이상의 Class가 mapping Dictionary 에 존재하면 t_class 에 할당. 아니면 None 으로 변경 
                    t_class = t_class_map[mul_class] if mul_class in t_class_map else None

                    if t_class != None:                                         
                        #mapping 되어있는 Multi Class 일 경우.  
                        #print("cls : {} box :{}".format(t_class, ybox))               
                        write_cls_bbox(t_class, bbox, fp)
                    else:                                                       
                        #mapping 되어있지 않은 Multi Class일 경우. 
                        #일단 넘어가고 나중에 아무것도 적혀있지 않은 파일들은 다 삭제.
                        continue

                else:                           
                    #Signal 이 없을때
                    t_class = t_class_map['unknown'] 
                    if t_class != None:                    
                        #print("cls : {} box :{}".format(t_class, ybox))               
                        write_cls_bbox(t_class, bbox, fp)

    except Exception as e:
        print(e)
        ...

def delete_empty_files(path='./'):
    '''
    내용 없는 파일 삭제
    '''
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.getsize(filepath) == 0:
            #파일의 size 가 0 인건
            os.remove(filepath) # 삭제
            print(f'{filepath} has been removed')


def main(YOLO_PATH='./'):

    os.chdir(YOLO_PATH) #경로 yolo_path 로 변경
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    TRAINING_LABEL_PATH = './dataset/labels_json/train/'
    VALIDATION_LABEL_PATH = './dataset/labels_json/val/'

    #YoloStyle 변환전에 데이터 전처리후 변환 해야함. 
    #TRAINING_YOLO_LABEL_PATH = './dataset/labels/train/'
    #VALIDATION_YOLO_LABEL_PATH = './dataset/labels/val/'

    #따라서 일단 JSON 파일에서 BBOX 와 CLS 만 추출.
    TRAINING_TXT_LABEL_PATH = './dataset/labels_txt/train/'
    VALIDATION_TXT_LABEL_PATH = './dataset/labels_txt/val/'



    # 매칭되는 이미지의 크기 직접 구해서 넣으려 했는데 json 파일에 이미지 크기 정보도 같이 있어서 그거 씀 
    #TRAINING_IMG_PATH = './dataset/images/train/'         
    #VALIDATION_IMG_PATH = './dataset/images/val/'

    label_path_map ={
        TRAINING_LABEL_PATH : TRAINING_TXT_LABEL_PATH,
        VALIDATION_LABEL_PATH : VALIDATION_TXT_LABEL_PATH
    }
    
    for source_label_path, yolo_label_path in label_path_map.items():

        files = os.listdir(source_label_path)
        try:
            #0. json 파일 한개씩 read
            for json_file_name in files:
                file_path = os.path.join(source_label_path, json_file_name)
                
                #1. 파일 내용 json read
                with open(file_path, "r") as str_json:
                    label_json = json.load(str_json)

                #"image":{"filename":"s01000113.jpg","imsize":[1280,720]}
                image_inf = label_json['image']
                
                for obj in label_json['annotation']:
                
                    if obj['class'] == 'traffic_light': # 2. class 가 traffic light 일 경우만 txt 파일에씀.
                        #generate_yolo_label(obj, json_file_name, image_inf, yolo_label_path)
                        generate_label_file(obj, json_file_name, image_inf, yolo_label_path)

                    elif obj['class'] == 'traffic_information': # 유고정보 (cone, pothole, constructionm, unknone) 이라는데 같이 학습 시켜볼까?
                        continue

                    elif obj['class'] == 'traffic_sign': # class 가 traffic sign 일 때는 무시. (기존 traffic sign 데이터셋이 더 상세하게 구성 되어있음) 
                        continue    

                    else:
                        continue

            print(len(files))

            '''
            # txt 파일에 내용이 없을 경우 삭제. 즉 데이터셋중 Class 에 정의 되지 않았던 객체만 있는 이미지일 경우 삭제함. 
            # (만약 class 에 있는게 있으면 하나라도 기록이 되기 때문에 삭제 안됨) 
            '''
            delete_empty_files(yolo_label_path)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    
    YOLO_PATH = '/home/woojin/_traffic_sig/yolov5/'
    print("path",YOLO_PATH)
    main(YOLO_PATH)

    '''
    path
    |-- yolov5
        |-- dataset_util
            |-- *[this_file]
        |-- dataset
            |--images
                |-- train
                |-- val
            |--labels
                |-- train
                |-- val
    '''





"""
def copy_file(obj, json_file_name, image_inf):
    copied = ''
    try:
        for attr in obj['attribute']:
            signal = [key for key, value in attr.items() if value == 'on']
            if(len(signal) > 1):
                if copied != json_file_name:
                    shutil.copy(SOURCE_LABEL_PATH+json_file_name, PRE_PROCESSING_DIR+"mul_attr/")
                    shutil.copy(SOURCE_IMG_PATH+image_inf['filename'], PRE_PROCESSING_DIR+"mul_attr/")
                    copied = json_file_name

    except Exception as e:
        print(e)
"""
"""
JSON 파일 예시

{"annotation":
[{"shape":"circle","color":"red","kind":"normal",
  "box":[624,133,640,149],"text":"0","type":"restriction","class":"traffic_sign"},

   # 신호 감지 못함 (unknown) 
  {"light_count":"unknown","box":[774,138,787,157],
  "attribute":[{"red":"off","green":"off","x_light":"off","others_arrow":"off","yellow":"off","left_arrow":"off"}],
  "type":"unknown","class":"traffic_light","direction":"vertical"},

   # 신호 감지 1개 
  {"light_count":"3","box":[589,139,611,145],
  "attribute":[{"red":"off","green":"on","x_light":"off","others_arrow":"off",
  "yellow":"off","left_arrow":"off"}],"type":"car",
  "class":"traffic_light","direction":"horizontal"},

   # 여러개 신호 감지
  {"light_count":"3","box":[349,49,389,63],
  "attribute":[{"red":"off","green":"on","x_light":"off"
  ,"others_arrow":"off","yellow":"off","left_arrow":"on"}],"type":"car",
  "class":"traffic_light","direction":"horizontal"}],

  "image":{"filename":"s01802648.jpg","imsize":[1280,720]}}

annoation : [...] 
ㄴ [ class : traffic light ], or [class : traffi_sign -> 무시 ] or [class : traffic_information -> 무시]
     ㄴ attribute
        ㄴ red(on/off) , green(on/off) , yellow(on/off) , x_light(on/off) , others_arrow(on/off) , left_arrow(on/off) 

direction : vertical or horizontal or None. -> 이것까지 고려할 경우 Class 가 너무 많아져 뺌 (주로 Vertical 은 보행자 신호, horizaontal 은 차량 신호등으로 구분한것 같은데 없는것도 있어서 애매함.)
light_count : 대부분 1 ~ 4개 사이로 나옴. 1은 가변차로 신호등에서 주로 보이고 2는 보행자, 3~4는 차량신호등에서 주로 나오는데 잘 안보이는 신호등은 정확하지 않아서 뺌. 
"""
