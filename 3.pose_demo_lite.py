"""
树莓派运行：tflite人体检测+自定义关键点检测模型
"""

# 导入相关包
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


"""
采集训练素材
保存姿态图片，以及
"""
import cv2
import numpy as np

import time

class EnpeiPoints:
    """
    获取人体Pose关键点
    """
    def __init__(self):
        # 加载自定义关键点检测模型
        self.interpreter = tflite.Interpreter(model_path="./weights/custom_resnet_128/enpei_pose_resnet_128.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
       


    def getFramePose(self,frame):
        """
        获取关键点
        """
        img_input = self.load_images(frame)
        img_input = img_input.reshape(1,128,128,1)

        # result = self.pose_model.predict(img_input)
        img_input = np.array(img_input,dtype=np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        result = output_data
        print(result)


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
        # 加载自定义目标检测模型
        self.interpreter = tflite.Interpreter(model_path="./weights/hub/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        
        # 加载摄像头
        self.cap = cv2.VideoCapture(0)

        # 画面宽度和高度
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 关键点检测
        self.keypoints_model = EnpeiPoints()

    def detect(self):
        
        # 帧数
        
        start_time = time.time()
        while True:
            ret,frame = self.cap.read()

            if frame is None:
                break
            # 转为RGB
            img_cvt = cv2.resize(frame,(300,300))
            img_cvt = cv2.cvtColor(img_cvt,cv2.COLOR_BGR2RGB)
            img_input = img_cvt.reshape(1,300,300,3)

            # img_input = np.array(img_input,dtype=np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], img_input)

            self.interpreter.invoke()

            detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])
            detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            
            for index,score in enumerate(detection_scores[0]):
                if score > 0.5:

                    class_ = detection_classes[0][index]

                    if class_ == 0:
                        box = detection_boxes[0][index]
                    
                        t,l,b,r = box
                        l,r =  int(l*self.frame_w),int(r*self.frame_w)
                        t,b =  int(t*self.frame_h),int(b*self.frame_h)
                        cv2.rectangle(frame, (l,t), (r,b), (0,255,0),5)

                        # 裁剪
                        frame_crop = frame[t:b,l:r]
                        
                        # 关键点检测
                        print(t,b,l,r)
                        if t>0 and b >0 and l > 0 and r> 0:
                            points = self.keypoints_model.getFramePose(frame_crop)

                            img_h,img_w = b-t,r-l
                            for x,y in points:
                                x = int(x*img_w) +l
                                y = int(y*img_h) + t
                                cv2.circle(frame,(x,y),10,(255,0,255),-1)
                            

            now = time.time()
            fps_time = now - start_time
            start_time = now

            fps_txt =round( 1/fps_time,2)
            cv2.putText(frame, str(fps_txt), (50,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)
            cv2.imshow('demo',frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            

        self.cap.release()
        cv2.destroyAllWindows()


plate = Pose_detect()            
plate.detect()


