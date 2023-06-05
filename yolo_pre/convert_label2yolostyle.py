import json 
import os
import shutil

SOURCE_LABEL_PATH = '../data/Validation/label/'
SOURCE_IMG_PATH = '../data/Validation/image/'
YOLO_LABEL_PATH = '../yolov5/content/dataset/valid/'
PRE_PROCESSING_DIR = '../data_pre/Validation/'

def convert_yolo_style(xyxy:list, img_size:list) -> list:
    '''
    Labeling 된 BBOX의 x1, y1, x2, y2 좌표를 YoloStyle 로 변환 하는 코드
    xywh 라면 밑에 w와 h 계산할 필요 없이 바로 넣는다.
    '''
    
    (x1, y1, x2, y2) = xyxy
    img_w, img_h = img_size
    ## 좌표를 이미지 비율에 맞춰 변환하기 위한 dw, dh
    dw = 1./img_w
    dh = 1./img_h
   
    x = float(x1)
    y = float(y1)
    w = float(x2)-float(x1)     #xyxy2xywh
    h = float(y2)-float(y1)
    
    # x와 y 의 중심좌표 업데이트 
    x = (x + x + w)/2.0 
    y = (y + y + h)/2.0
    w = w
    h = h
    
    # 이미지 크기에 대한 비율로 변환
    x = round(x*dw, 6)
    w = round(w*dw, 6)
    y = round(y*dh, 6)
    h = round(h*dh, 6)

    return [x, y, w, h]


def write_yolo_label(cls:str, y_xywh:list, fp, save_path='./'): 
    '''
    Yolo Label txt 파일로 write 하는 함수

    밖에서 파일포인터 열고 포인터를 argument로 받아 여기서 작업 후 
    끝나면 밖에서 다시 닫는 식으로 사용해야함.
    (함수 들어올때마다 파일을 열면 속도저하 문제 발생)
    '''
    yx, yy, yw, yh = map(str,y_xywh)
    fp.write(' '.join([cls, yx, yy, yw, yh]))
    fp.write('\n')
    print(" {} : write Success".format(str(fp)))


def generate_yolo_label(obj:dict, json_file_name:str, image_inf:dict) -> str: 
    '''
    JSON 파일의 {Attribute ('red':'on/off','yellow':'on/off','green':'on/off','left_arrow':'on/off','x_light':'on/off','others_arrow':'on/off')}
    에 on/off 유무에 따라 Class 를 정하고 Mapping 후 Yolostyle로 변경하여 txt 파일로 생성하는 함수.

    이때 Attribute 에 on 으로 표기된게 2개 이상일 경우, Class 별 개수와 의미에 따라 삭제 혹은 병합하도록 하였음 (2개이상 Class 는 Loop의 이슈사항 탭 참조)
    ex1) left_arrow, red_left_arrow 는 둘다 좌회전을 의미 하기 때문에 '4' Class 로 Mapping. 
    ex2) red_green, red_gereen_yellow, red_green_others_arrow... 같은 데이터셋이 가끔 나오는데 의미를 알 수 없기 때문에 삭제.

    # 데이터셋의 갯수가 적은 중복클래스들을 전부 학습시키는건 데이터 불균형 문제로 당연히 안되고, 그렇다고 저걸 no-mapping 클래스라고 묶어서 하나로 학습시킬 경우 다양한 문제가 발생할 가능성이 높다
    #     1. no-mapping 클래스 범주로 분류하는 이미지들이 명확한 정보 제공을 하지 않을 것 (어떤 클래스에도 속하지않음이라는 모호한 개념을 표현하기때문), 
    #        ㄴ 또한 제대로된 특징이 없기 때문에 제대로 학습 못할 것
    #     2. 모델에서 추론을 진행할때 조금이라도 애매모호한 경우 no-mapping 클래스를 과도하게 사용 할 것으로 예상됨. -> 정확도에 영향을 끼침
    #     3. 이후 추가로 학습이 어려움. 

    따라서 여러개 나온 클래스중 의미가 같은걸 최대한 묶어서 학습, 수량이 적은 데이터셋은 학습 안하고 제외, 필수적인 데이터셋은 Augmentation이라도 해서 학습시킴
    '''
    #Class Mapping Dictionary
    t_class_map = {'unknown':'0',                                              
                   'red':'1',
                   'yellow':'2','red_yellow':'2','green_yellow':'2',
                   'green':'3',
                   'left_arrow':'4','red_left_arrow':'4',
                   'green_left_arrow':'5',
                   'x_light':'6','others_arrow':'7'}
    t_class = None

    try:
        #txt file open
        with open(YOLO_LABEL_PATH+json_file_name[:-4]+"txt", "a") as fp:

            for attr in obj['attribute']:
                
                traffic_signal = [key for key, value in attr.items() if value == "on"] #str
                ybox = convert_yolo_style(obj['box'], image_inf['imsize']) #list

                if(len(traffic_signal) == 1):   #traffic_signal이 하나일때
                    t_class = t_class_map[''.join(traffic_signal)]
                    if t_class != None:     
                        #print("cls : {} box :{}".format(t_class, ybox))               
                        write_yolo_label(t_class, ybox, fp)              

                elif(len(traffic_signal) > 1):  
                    #만약 attribute 가 동시에 2개 이상 있다면 2가지 값을 합쳐서 클래스를 만듬. (ex. red, left_arrow => red_left_arrow : 0)
                    mul_class = '_'.join(traffic_signal)                    

                    # 2개이상의 Class가 mapping Dictionary 에 존재하면 mapping된 클래스를 t_class 변수에 . 아니면 None 으로 변경 
                    t_class = t_class_map[mul_class] if mul_class in t_class_map else None

                    if t_class != None:                                         
                        #mapping 되어있는 Multi Class 일 경우.  
                        #print("cls : {} box :{}".format(t_class, ybox))               
                        write_yolo_label(t_class, ybox, fp)
                    else:                                                       
                        #mapping 되어있지 않은 Multi Class일 경우. 
                        #일단 넘어가고 나중에 아무것도 적혀있지 않은 파일들은 다 삭제.
                        continue

                else:                           
                    #Signal 이 없을때
                    t_class = t_class_map['unknown'] 
                    if t_class != None:                    
                        #print("cls : {} box :{}".format(t_class, ybox))               
                        write_yolo_label(t_class, ybox, fp)

    except Exception as e:
        print(e)
        ...


def delete_empty_files(path=YOLO_LABEL_PATH):
    '''
    내용 없는 파일 삭제
    '''
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.getsize(filepath) == 0:
            #파일의 size 가 0 인건
            os.remove(filepath) # 삭제
            print(f'{filepath} has been removed')


def main():

    files = os.listdir(SOURCE_LABEL_PATH)
    try:
        #0. json 파일 한개씩 read
        for json_file_name in files:
            file_path = os.path.join(SOURCE_LABEL_PATH, json_file_name)
            
            #1. 파일 내용 json read
            with open(file_path, "r") as str_json:
                label_json = json.load(str_json)

            #print("json_file_name : {}".format(json_file_name))SOURCE_LABEL_PATH
            #"image":{"filename":"s01000113.jpg","imsize":[1280,720]}
            image_inf = label_json['image']
            
            for obj in label_json['annotation']:
              
                if obj['class'] == 'traffic_light': # 2. class 가 traffic light 일 경우만 txt 파일에씀.
                    #copy_file(obj, json_file_name, image_inf)
                    generate_yolo_label(obj, json_file_name, image_inf)

                elif obj['class'] == 'traffic_information': # 유고정보 (cone, pothole, constructionm, unknone) 이라는데 같이 학습 시켜볼까?
                    #print(match_file, img_size)
                    continue

                elif obj['class'] == 'traffic_sign': # class 가 traffic sign 일 때는 무시. (기존 traffic sign 데이터셋이 더 상세하게 구성 되어있음) 
                    continue    

                else:
                    continue

        print(len(files))

        # txt 파일에 내용이 없을 경우 삭제. 즉 데이터셋중 Class 에 정의 되지 않았던 객체만 있는 이미지일 경우 삭제함. 
        # (만약 class 에 있는게 있으면 하나라도 기록이 되기 때문에 삭제 안됨) 
        delete_empty_files(YOLO_LABEL_PATH)


    except Exception as e:
        print(e)

if __name__ == "__main__":
    print("start")
    main()



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
