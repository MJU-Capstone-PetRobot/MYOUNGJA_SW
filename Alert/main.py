from twilio.rest import Client
import serial
import json
import googlemaps
import time

# Twilio
# 계정 token 입력
user_phone_number = Phone_Number_user
twilio_phone_number = Phone_Number_Twilio
account_sid = Twilio_API_sid_KEY
auth_token = Twilio_API_token_KEY
client = Client(account_sid, auth_token)

# GPS 위치 가져오기
file_path = "./gps_info.json"
with open(file_path, "r") as json_file:
    json_data = json.load(json_file)
    latitude = json_data["gps"][0]["latitude"]
    longitude = json_data["gps"][0]["longitude"]

# Google Maps
YONGIN_LAT = 37.24278
YONGIN_LONG = 127.17889
API = Googlemap_API_KEY  # API 값

# 위도 경도 -> 지번 주소로 변경 // 역지오코드
gmaps = googlemaps.Client(key=API) # api key
reverse_geocode_result = gmaps.reverse_geocode((latitude, longitude), language='ko')
# reverse_geocode_result = gmaps.reverse_geocode((YONGIN_LAT, YONGIN_LONG), language='ko')
gps = reverse_geocode_result[1]["formatted_address"]

message_send = 1 # 메세지 송신 트리거

# 화재 감지 코드
if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
    ser.flush()
    while message_send == 1:
        if ser.in_waiting > 0 :
            line = ser.readline().decode('utf-8').rstrip()
            s_line = int(line)
            if s_line > 200:
                message_send *= -1

            if message_send < 0: # 실험을 위해서 200을 기준으로 설정
                message = client.messages \
                    .create(
                    body=f"위험 위험!!, 할머니가 위험해요. 종류 : 화재 발생. 위치는 {gps}",
                    from_= twilio_phone_number,
                    to= user_phone_number
                )
            time.sleep(1000000)
