from tkinter import *
from PIL import Image, ImageTk
import subprocess
import pymysql

# 데이터베이스 연결 설정
conn = pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='1234',
    database='Pose',
    port=3306
)

# 파일 경로 설정
file1_path = "자세 감지\\PoseDetection.py"
file2_path = "눈깜빡임\\EyeBlink.py"

# Tkinter 창 생성
root = Tk()
root.geometry("600x400")
root.title("Balance POSE ProGram")

# 이미지 로드 및 크기 조정
image = Image.open("image.png")
image = image.resize((400, 300))
background_image = ImageTk.PhotoImage(image)

# 배경 이미지 표시
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# 파일 실행 함수
def run_file1():
    subprocess.call(["python", file1_path])

def run_file2():
    subprocess.call(["python", file2_path])

# 파일 실행 버튼
button1 = Button(root, text="파일 1 실행", command=run_file1, width=15, height=2)
button2 = Button(root, text="파일 2 실행", command=run_file2, width=15, height=2)

# 버튼 위치 설정
button1.place(x=100, y=350)
button2.place(x=400, y=350)

root.mainloop()
