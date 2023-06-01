#JRP Manager 트럭 데이터 수집 매크로
from subprocess import Popen, PIPE
import sys
import schedule
import time, datetime
import pyautogui as macro
import psutil
import gc
import os

#pipe = Popen("dir",shell=True, stdout=PIPE)
#PIPE : 한 프로그램의 출력을 다른 프로그램의 입력으로 연결해주는 연결 프로그램
#pipe = Popen("dir",shell=True, stdout=PIPE)

import sys
from PyQt5.QtWidgets import *

class overclicks(Exception): #예외 Exception Class 를 상속 받아 새로운 예외 생성 
    def __init__(self):
        super().__init__('img 를 찾을 수 없는 예외 발생')

def exit_program():
    #JRP Manager 종료
    for proc in psutil.process_iter():
        if proc.name() == 'JrpManager.exe':
            proc.kill()

    print("JRP Manager 종료")
    gc.collect() #참조중인 메모리 정리
    sys.exit(0)  #현재 프로그램 종료


if __name__ == '__main__':

    parent_process_id = os.getpid()
    pre_dir_size = -1
    now_dir_size = 0

    def get_dir_size(path='.'):
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
        return total

    def check_dir_volume(dir_name):
        # 다음페이지 넘어가기전 디렉토리 용량체크를 통해 넘어가도 되는지 확인.
        global pre_dir_size, now_dir_size
        path = f"F:\_JRP_Data\{dir_name}"
        print(path)
        while True:
            time.sleep(2)
            now_dir_size = get_dir_size(path)
            print("scan_directory")
            print(f"pre size : {pre_dir_size}")
            print(f"now size : {now_dir_size}")
            if(pre_dir_size != now_dir_size):
                pre_dir_size = now_dir_size
            else:
                print("download Complete?")
                time.sleep(1)
                return 1

    # 매크로 처음 시작할때 폴더 생성하는 함수
    def make_dir(dir_name):
        # 디렉토리 생성
        path = "F:/_JRP_Data/"
        directory = path+dir_name
        os.mkdir(directory)

        # 생성된 디렉토리 확인
        if os.path.isdir(directory):
            print(f'{directory} 디렉토리가 생성되었습니다.')

    def make_new_dir_macro(dir_name):
        while True:

            #chk
            check('./click_img/_chk_folder.PNG')
            time.sleep(2)

            # 새폴더 생성
            #work_in_list(['make_new_dir.png'], 1)
            macro.keyDown('alt')
            time.sleep(0.2)
            macro.press('m')
            macro.keyUp('alt')
            time.sleep(2)

            #chk
            check('./click_img/_chk_newfolder2.PNG')
            time.sleep(2)

            macro.write(str(dir_name))
            print("write ",str(dir_name))
            return dir_name

    def save_jrp_img(dir_name, delay = 5):
        clk_li = ['My_book_btn.PNG','save_chk.PNG','_jrp_data.png']
        while True:
            # ctrl+shift+p  (JRP Manager 단축키 전체 이미지 저장)
            # macro.press(['ctrl', 'shift', 'p'])  
            try:
                time.sleep(1)
                work_in_list(['chk_jrp.PNG'])
                time.sleep(delay)
                macro.keyDown('ctrl')
                time.sleep(0.25)
                macro.keyDown('shift')
                time.sleep(0.25)
                macro.press('p')
                macro.keyUp('ctrl')
                macro.keyUp('shift')
                print("push ctrl shift p")
                time.sleep(delay)

                # 뒤로 갈수록 ImageNum 뜨는 속도가 느려져서 이 창이 떴는지 한번 체크한뒤 넘어가야 될것 같다.
                # 안떴으면 다시 ctrl+shift+p.
                work_in_list(['chk_imageNum.png'])
                time.sleep(delay)

                # 4 (후면 이미지)
                macro.write('4')
                time.sleep(delay)
                macro.press('enter')
                time.sleep(delay)
                
                # 이미지 저장 경로 설정 > MyBook > _jrp_manager
                work_in_list(clk_li, c=2)
                time.sleep(delay)
                make_new_dir_macro(dir_name)
                time.sleep(delay)
                macro.press('enter')
                return 1
            except overclicks as e:
                print("예외 발생 save_jrp_img() 재실행", e)
                continue

    def work_in_list(exec_li, c=1):
        #loop block
        for eb in exec_li:
            click('./click_img/'+str(eb), clicks=c)
            print("click",eb)
            time.sleep(1)
            ...
    
    def click(path = '.', clicks = 1):
        #100% 만 클릭하기때문에 가끔 못찾을때가 있어서 클릭함수 개선
        count = 0
        while True:
            clk_loc = macro.locateOnScreen(path, confidence=0.98)
            if clk_loc != None:
                print("clk_loc :", clk_loc)
                macro.click(clk_loc)
                return clk_loc
            else:
                count += 1 
                print(f"{path} 로딩중... ")
                time.sleep(3)
                if count == 10:
                    raise overclicks #10번 이상 click 이 안되면 overclicks 라는 예외를 발생시킴                     ...
                else:
                    continue

    #delay 줘서 하니까 로딩이 느려져서 안되는 경우가 빈번히 발생
    #클릭은 하지 않고 다음 단계로 잘 넘어가는지 체크만 하는 함수

    def check(path):
        count = 0
        while True:
            chk_loc = macro.locateOnScreen(path, confidence=0.98)
            print("check ",path)
            if chk_loc != None:
                return True
            else:            
                count += 1 
                print(f"{path} check 실패... ")
                time.sleep(3)
                if count == 10:
                    raise overclicks #10번 이상 check가 안되면 overclicks 라는 예외를 발생시킴                     ...
                else:
                    continue

    #버튼클릭시 동작
    def do(dir_name, page_num):
        time.sleep(2)
        #1) 파일 생성
        make_dir(dir_name)
        time.sleep(2)

        for i in range(int(page_num)):
            print(f"{i+1}page")
            #2) ctrl+shift+p, 4, 새파일 생성 저장
            save_jrp_img(dir_name)                
            time.sleep(2)
            macro.press('enter')                    # 폴더이름 중복확인 엔터
            time.sleep(2)
            work_in_list(['enter.png'], 1)          #저장 시작
            check_dir_volume(dir_name)      
            time.sleep(2)
            work_in_list(['next_page1_btn.PNG'])    # 다음페이지 클릭

        print("파일 저장 완료 프로그램 종료")
        exit_program()
        sys.exit(0)
    
    #간단한 ui 구성
    def qt_widget():
        app = QApplication(sys.argv)
        window = QWidget()
        window.setWindowTitle("macro test")
        
        #main_layout batch start
        main_layout = QVBoxLayout()
        label = QLabel("JRP_Manager에서 원하는 날짜로 이동한뒤 총 페이지 개수를 입력후 버튼을 누르세요")
        main_layout.addWidget(label)
        
        #input layout batch start
        input_layout1 = QHBoxLayout()
        dir_label = QLabel("디렉토리 이름 : ")
        input_layout1.addWidget(dir_label)
        scf_dir_name = QLineEdit()
        input_layout1.addWidget(scf_dir_name)
    
        input_layout2 = QHBoxLayout()
        page_num_label = QLabel("총 페이지수 : ")
        input_layout2.addWidget(page_num_label)
        scf_page_num = QLineEdit()
        input_layout2.addWidget(scf_page_num)
        #input layout batch end

        main_layout.addLayout(input_layout1)
        main_layout.addLayout(input_layout2)
        
        start_btn = QPushButton('Start')
        start_btn.clicked.connect(lambda: do(scf_dir_name.text(), scf_page_num.text()))
        main_layout.addWidget(start_btn)
        #main_layout batch end

        window.setLayout(main_layout)
        window.show()
        app.exec_()
        
    def job():
        jrp_ch = 0
        clk_li = ['exit_btn.PNG','login_btn.PNG','view_history_btn.PNG','yesterday_btn.PNG']
        #JRP_manager 실행여부 체크
        for process in psutil.process_iter():                           
            if 'JrpManager.exe' in process.name():
                jrp_ch = 1
                break

        if not jrp_ch: #JRP Manager 실행 안되어있을 경우
            print("JRP Manage 실행. 원하는 날짜로 이동하세오")
            Popen("C:\Program Files (x86)\JrpManager\JrpManager.exe")   #JrpManager.exe 실행
            #1) 팝업닫기, 로그인, 이력조회, 어제버튼 누르는 매크로 실행
            work_in_list(clk_li, 1)
            time.sleep(2)
            
        else:
            #pyqt
            print("qt exec")
            qt_widget()
        
    #start main
    macro_process_id = os.getpid()    
    #schedule.every().day.at("06:00").do(job) #매일 아침 6시마다 job 함수 실행.
    schedule.every(1).seconds.do(job)   # 1초마다 job 실행
    #cnt = 3
    while True:                 
        #print("Current Mouse Position", macro.position())
        schedule.run_pending()          #5초에 한번씩 스케쥴을 체크. 실행중이지 않으면 실행. 
        #print(cnt)
        #macro.press('.')               #보안프로그램때문에 자꾸 화면 꺼져서 꺼지지 말라고 넣음...
        time.sleep(5)
        #cnt = cnt -1 if cnt-1 else 3

    
