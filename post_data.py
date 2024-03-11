import requests,json
#192.168.137.1 
#192.168.115.167

#PC의 mariaDB서버로 요청 데이터 전송하여 저장시키기
URL = 'http://192.168.137.1:8000/pose'

def http_post_data(data):
    while True:
        btn_data = {
            'incorrectpose': data[0], #보낼 데이터
            'normalpose' : data[1],
            'correctpose' : data[2],
            'blinkdata' : 13
        }
        try:
            res = requests.post(URL, json=btn_data)
            print(res.status_code)
            res_data = json.loads(res.text)
            print(res_data)
        except:
            print("connection failed")
        break

# 데이터 보낼때 실행
#     while True:
#     ex) htdata = sensor.temphumidity_measure()
#         print(htdata)
#         post_data.http_post_data(htdata)

#         time.sleep(1)
