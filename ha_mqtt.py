"""
MQTT协议控制Home assistant
"""
from paho.mqtt import client as mqtt_client
import threading

import time
import random

class HaMqtt:
    """
    1.发送MQTT协议给home assistant
    2.播放动作配乐
    """
    def __init__(self):


        # mqtt配置
        self.broker = '192.168.1.165'
        self.mqtt_port = 1883
        self.mqtt_topic = "enpei/action/play"

        self.client_id = f'python-mqtt-{random.randint(0, 1000)}'
        # 用户名密码
        self.username = 'enpei'
        self.password = '3020484'

        # 初始化
        self.mqtt_client = self.connect_mqtt()
        self.mqtt_client.loop_start()

       


    def connect_mqtt(self):
        """
        连接MQTT服务器
        """
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("连接成功!")
            else:
                print("连接失败：code %d\n", rc)

        client = mqtt_client.Client(self.client_id)
        client.username_pw_set(self.username, self.password)
        client.on_connect = on_connect
        client.connect(self.broker, self.mqtt_port)
        return client


    def sendRemoteCommand(self, msg):
        """
        发送指令
        """
        result = self.mqtt_client.publish(self.mqtt_topic, msg)
        status = result[0]

        if status == 0:
            print('成功发送指令')
        else:
            print(f"发送指令失败")

    def test(self):
        """
        测试指令
        """
        index = 0
        while True:
            msg = 'from_action_{}'.format(index)
            self.sendRemoteCommand('switch')
            time.sleep(5)
            index+=1


# hamqtt = HaMqtt()
# hamqtt.test()
