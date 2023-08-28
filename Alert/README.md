# Alert

화재 감지하여 119에 문자메세지 전송

## 개발환경
- Computer : Orange Pi 5B
- OS : Ubuntu 22.04
    - python3.10
    - python package
        - Twilio API : `sudo pip3 install twilio`
            - Twilio Account ID : companion.robot.mju@gmail.com
            - Twilio Account PW : 3288f#sdjf32Fsdjk
            - Account SID : ACede10e87e024116f5ccef5f41323d84b
            - Auth Token : 9b5738217084cc061ce7acc5702d3408
            - My Twilio phone number : +13203772706
        - Serial : `sudo pip3 install pyserial`
        - Google Map : `sudo pip3 install googlemaps`
- Sensor
    - MQ-7 연기 센서
    - GPS

## Alert Function 
- MQ-7 연기 센서를 이용한 화재 감지
- GPS를 이용한 화재 발생 위치 파악
- 화재 발생 시 119에 신고 문자 전송

## Log
- (230409) Alert mail & message by Python
- (230506) 라즈베리파이 이용 기준, 연기 감지 센서를 통해서 데이터를 받아 gps 상 위치까지 확인하여 위기 상황 메세지를 전송하는 기능 구현(실제 실험 필요) 
- (230510) MQ-7 연기 센서 코드 제작 완료 / GPS는 아직...
- (230511) MQ-7 + GPS + TWILIO 코드 최종 완성 / 실험 통과!
- (230522) 낙상 감지 후 메세지 전송  추가 / 낙상 감지 기능, 라즈베리파이에서 실행 가능한지 추가 실험 필요
