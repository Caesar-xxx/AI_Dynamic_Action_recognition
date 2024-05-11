"""
Windows或树莓派运行：录制动作视频序列
抽出关键点
"""

import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite


class HumanKeypoints:
    """
    获取人体Pose关键点
    """
    def __init__(self):
        
        # 加载自定义关键点检测模型
        self.interpreter = tflite.Interpreter(model_path="./weights/hub/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def getFramePose(self,image):
        """
        获取关键点
        """
        # 转为RGB
        img_cvt = cv2.resize(image,(192,192))
        img_cvt = cv2.cvtColor(img_cvt,cv2.COLOR_BGR2RGB)
        img_input = img_cvt.reshape(1,192,192,3)


        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)

        self.interpreter.invoke()

        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 1,1,17,3 
        #  [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]).
        # 只需要以下
        # left shoulder: 5
        # left shoulder:6
        # left elbow: 7
        # right elbow: 8
        # left wrist: 9
        # right wrist: 10
        #
        keypoints_with_scores = keypoints_with_scores[0][0][5:11]

        return keypoints_with_scores

    


class ActionRecord:
  def __init__(self):
    # 加载关键点检测模型
    self.human_keypoints = HumanKeypoints()

  def record(self):

    cap = cv2.VideoCapture(0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), 15, (frame_w,frame_h))

    while True:
      ret,frame = cap.read()
      videoWriter.write(frame)
      # 获取该帧特征关键点
      keypoints = self.human_keypoints.getFramePose(frame)
      # 显示
      for y,x,score in keypoints:
          x = int(x * frame_w)
          y = int(y * frame_h)
          cv2.circle(frame,(x,y),10,(0,255,0),-1)

      # frame = cv2.flip(frame,1)
      cv2.imshow('MediaPipe Hands', frame)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()



action = ActionRecord()
action.record()