"""
Windows运行：YOLOv5人体检测+自定义关键点检测模型（RESNET 128）
"""

# 导入相关包
import cv2
import numpy as np
import tensorflow as tf

"""
采集训练素材
保存姿态图片，以及
"""
import torch
import cv2
import numpy as np

import time

class EnpeiPoints:
    """
    获取人体Pose关键点
    """
    def __init__(self):
        # 加载自定义关键点检测模型
        self.pose_model = tf.keras.models.load_model('./weights/custom_resnet_128/best_model_resnet_128.hdf5')

    def getFramePose(self,frame):
        """
        获取关键点
        """
        img_input = self.load_images(frame)
        img_input = img_input.reshape(1,128,128,1)
        result = self.pose_model.predict(img_input)
        kepoints = result[0].reshape((-1,2))

        return kepoints


    def load_images(self,img):
        """
        缩放、灰度图、归一化
        """
        img = cv2.resize(img,(128,128))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img/255
        img = np.reshape(img, (128,128,1))
        return img




class Pose_detect:

    def __init__(self):
        
        # 加载模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./weights/yolov5/yolov5m.pt',source='local')  # local repo
        # 置信度阈值
        self.model.conf = 0.4
        # 加载摄像头
        self.cap = cv2.VideoCapture(0)

        # 画面宽度和高度
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 关键点检测
        self.keypoints_model = EnpeiPoints()

    def detect(self):
        
        # 帧数
        frame_index = 4000
        start_time = time.time()
        while True:
            ret,frame = self.cap.read()

            if frame is None:
                break
            # 转为RGB
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 推理
            results = self.model(img_cvt)

            pd = results.pandas().xyxy[0]
            person_list = pd[pd['name']=='person'].to_numpy()

            
            # 遍历每个人
            for person in person_list:
                l,t,r,b = person[:4].astype('int')

                frame_crop = frame[t:b,l:r]

                # 关键点检测
                points = self.keypoints_model.getFramePose(frame_crop)

                img_h,img_w = b-t,r-l
                for index,(x,y) in  enumerate(points):
                    x = int(x*img_w) +l
                    y = int(y*img_h) + t
                    cv2.putText(frame, str(index), (x-20,y-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
                    cv2.circle(frame,(x,y),10,(255,0,255),-1)
                
                cv2.rectangle(frame, (l,t), (r,b), (0,255,0),5)
                # 
            now = time.time()
            fps_time = now - start_time
            start_time = now

            fps_txt =round( 1/fps_time,2)
            cv2.putText(frame, str(fps_txt), (50,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)
            cv2.imshow('demo',frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            frame_index +=1

        self.cap.release()
        cv2.destroyAllWindows()


plate = Pose_detect()            
plate.detect()


