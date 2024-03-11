import cv2
import dlib
from scipy.spatial import distance
from datetime import datetime
import matplotlib.pyplot as plt

# 얼굴 인식기와 랜드마크 인식기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 깜빡임 비율 계산 함수
def eye_aspect_ratio(eye):
    # 눈의 수직 방향 눈꺼풀 간 거리 계산
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # 눈의 수평 방향 눈꺼풀 간 거리 계산
    C = distance.euclidean(eye[0], eye[3])

    # 눈 깜빡임 비율 계산
    ear = (A + B) / (2.0 * C)
    return ear

# 눈 깜빡임 비율에 대한 임계값 설정
EYE_AR_THRESH = 0.25

# 연속적으로 감지한 프레임 수와 눈 깜빡임 수 초기화
COUNTER = 0
TOTAL = 0

# 시간별 눈깜빡임 횟수 저장을 위한 리스트 초기화
times = []
blink_counts = []

# 비디오 스트림 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # 프레임을 회색으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray, 0)

    for face in faces:
        # 얼굴의 랜드마크 추출
        landmarks = predictor(gray, face)

        # 눈 깜빡임 비율 계산
        left_eye = []
        right_eye = []

        for n in range(36, 42):
            left_eye.append((landmarks.part(n).x, landmarks.part(n).y))
        for n in range(42, 48):
            right_eye.append((landmarks.part(n).x, landmarks.part(n).y))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # 눈 깜빡임 검출
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= 3:
                TOTAL += 1
            COUNTER = 0

        # 현재 시간 기록
        current_time = datetime.now().strftime("%H:%M:%S")
        times.append(current_time)
        blink_counts.append(TOTAL)

        # 눈 깜빡임 횟수 표시
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 프레임 표시
    cv2.imshow("Frame", frame)

    # 종료 단축키 설정
    if cv2.waitKey(1) == ord('q'):
        break

# 비디오 스트림 종료
cap.release()
cv2.destroyAllWindows()

# 시간별 눈깜빡임 횟수 막대 그래프 생성
plt.bar(times, blink_counts)
plt.xlabel('Time')
plt.ylabel('Blink Count')
plt.title('Blink Count over Time')
plt.xticks(rotation=45)
plt.show()
