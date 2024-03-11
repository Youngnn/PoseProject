import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import pymysql

# 데이터베이스 연결 설정
conn = pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='1234',
    database='Pose',
    port=3306
)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

blink_count = 0
prev_blink = False

blink_history = []
times = []  # 시간 정보를 저장할 리스트

def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return eyes

while True:
    ret, frame = cap.read()

    eyes = detect_eyes(frame)

    if len(eyes) == 0:
        if not prev_blink:
            blink_count += 1
            prev_blink = True
            print("눈 깜빡임 감지:", blink_count)

            # 눈을 감지했을 때에만 시간 정보를 추가
            times.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        prev_blink = False

    blink_history.append(blink_count)

    cv2.putText(frame, "Blink Count: {}".format(blink_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 데이터베이스에 눈 깜빡임 데이터 삽입
with conn.cursor() as cursor:
    for i, blink in enumerate(blink_history):
        if i < len(times):
            timestamp = times[i]
        else:
            # times 리스트에 대응하는 시간 정보가 없는 경우에는 현재 시간 사용
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        query = "INSERT INTO blink_data (blink_count, timestamp) VALUES (%s, %s)"
        values = (blink, timestamp)
        cursor.execute(query, values)

conn.commit()
conn.close()

# 막대 그래프로 눈 깜빡임 횟수 표시
plt.bar(range(1, len(blink_history) + 1), blink_history)
plt.axhline(y=20, color='r', linestyle='--', label='Drowsiness Threshold')
plt.xlabel('Frames')
plt.ylabel('Blink Count')
plt.title('Blink Count Chart')
plt.legend()
plt.show()