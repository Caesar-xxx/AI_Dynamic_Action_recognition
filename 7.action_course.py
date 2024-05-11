"""
Windows运行：tflite movenet关键点检测模型 + DTW动作识别
"""

# 导入相关包
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

import glob
import tqdm
import os
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import threading
from playsound import playsound
"""
Windows可能报错，需要将虚拟环境下Lib\site-packages下的playsound.py 修改：

第55行修改一下：
- command = ' '.join(command).encode('utf-16')
+ command = ' '.join(command)

第62行：
- '\n        ' + command.decode('utf-16') +
+ '\n        ' + command +
"""


class HumanKeypoints:
    """
    获取人体Pose关键点
    """
    def __init__(self):
        
        # 加载movenet关键点检测模型
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

    def getVectorsAngle(self,v1,v2):
        """
        获取两个向量的夹角，弧度
        cos_a = v1.v2 / |v1||v2|
        """
        if np.array_equal(v1,v2):
            return 0
        dot_product = np.dot(v1,v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)


        return np.arccos(dot_product/norm) 

        

    def getFrameFeat(self,frame):
        """
        获取特征
        1.获取17个关键点，取其中6个点
        2.计算线条之间的角度信息
        3.角度信息拼成一个特征向量
        """
        # 获取关键点
        pose_landmarks = self.getFramePose(frame)
        # 连接信息
        new_conn = [(0,1),(1,3),(3,5),(0,2),(2,4)]
       
        # 解析关键点
        p_list = [[landmark[1],landmark[0]] for landmark in pose_landmarks ]
        # 转为numpy，才能广播计算
        p_list = np.asarray(p_list)
        
        # 构造向量
        # conns关键点索引之间的连接关系，利用它来构造向量（终点坐标-起点坐标）
        vector_list = list(map(
            lambda con: p_list[con[1]] - p_list[con[0]],
            new_conn
        ))
        # 计算向量之间的角度，任意两个之间夹角
        """
        !warning: 此处应该可以优化，减少特征数量
        """
        angle_list = []
        for vect_a in vector_list:
            for vect_b in vector_list:
                angle = self.getVectorsAngle(vect_a,vect_b)
                angle_list.append(angle)
        return angle_list, pose_landmarks
        


class VideoFeat:
    """
    计算特征
    计算每一帧的特征
    计算每个视频的特征
    """
    def __init__(self):
        # 加载关键点检测模型
        self.human_keypoints = HumanKeypoints()
        # 加载动作训练集特征
        self.training_feat = self.load_training_feat()

        # 批次及阈值
        self.batch_size = 4
        self.threshold = 0.5



    def get_video_feat(self,filename):
        """
        读取单个视频，获取特征
        params:
            filename: str 文件名
        """
        cap = cv2.VideoCapture(filename)
        
        # 视频特征
        video_feat = []
        while True:
            ret,frame = cap.read()
            if frame is None:
                break
            
            # 获取该帧特征
            angle_list,results = self.human_keypoints.getFrameFeat(frame)

            # 追加
            video_feat.append(angle_list)

           
            # cv2.imshow('demo',frame)
        # 保存视频特征
        return video_feat

    def load_training_feat(self):
        """
        返回训练集的特征
        如果没有，则重新生成（读取所有训练数据集，存储为npz文件）
        """
        dataFile = './data/trainingData.npz'

        if os.path.exists(dataFile):
            with open(dataFile,'rb') as f:
                return np.load(f,allow_pickle='TRUE')
            
        filename = r'.\data\action_train\*\*.mp4'
        file_list = glob.glob(filename)
        training_feat = []
        for file in tqdm.tqdm(file_list,desc='训练数据集处理中') :
            action_name = file.split('\\')[3]
            video_feat = self.get_video_feat(file)
            training_feat.append([action_name,video_feat])

        # 转为numpy 数组
        training_feat = np.array(training_feat,dtype=object)
        # 写入文件
        with open(dataFile,'wb') as f:
            np.save(f,training_feat)
        
        return training_feat


    def calSimilarity(self,seqFeat):
        """
        计算序列特征与训练集之间的DTW距离
        给出最终预测动作名称
        """
        # 遍历训练集中特征
        dist_list = []
        for v_feat in self.training_feat:

            action_name,video_feat = v_feat
            distance, path = fastdtw(seqFeat, video_feat,dist=euclidean)
            dist_list.append([action_name,distance])

        # 转为numpy
        dist_list = np.array(dist_list,dtype=object)
        
        # 距离由低到高排序，并截取前batch_size个
        dist_list = dist_list[dist_list[:,1].argsort()][:self.batch_size]

        print(dist_list)

        # 获取排名第一的名称和距离
        first_key = dist_list[0][0]
        first_distance = dist_list[0][1]

        if first_distance > 100:
            print('未定义动作')
            return 'unknown' 

        # 计算该名称出现次数
        max_num = np.count_nonzero(dist_list[:,0] == first_key)

        # 计算排序第一个的，出现总数是否超过阈值
        if max_num / self.batch_size >= self.threshold:
            print('预测动作：{}，出现次数{}/{}'.format(first_key,max_num,self.batch_size))
            return first_key
        else:
            print('未定义动作')
            return 'unknown'

    def playVoice(self, fileName,mode):
        """
        播放音乐
        """
        playsound(fileName)
    
    def backPlay(self,fileName):
        """
        后台播放
        """
        t = threading.Thread(target=self.playVoice, args=(fileName,'voice'))
        t.start()

    def realTimeVideo(self):
        """
        实时视频流动作识别
        """
        cap = cv2.VideoCapture(0)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 录制状态
        record_status = False
        # 帧数
        frame_count = 0
        # 序列特征
        seq_feats = []

        triger_time = time.time()
        start_time = triger_time

        last_action = ''

        while True:
            ret,frame = cap.read()
            if frame is None :
                break

            # 获取该帧特征
            angle_list,results = self.human_keypoints.getFrameFeat(frame)
            
            if record_status:
                # 按R后等待3秒再识别
                if time.time() - triger_time >= 1:
                
                    if frame_count < 40:
                        # < 50 帧，录制动作
                        # 录制中红色
                        cv2.circle(frame,(50,50),20,(0,255,0),-1)
                        seq_feats.append(angle_list)
                            
                        frame_count +=1
                        # print('录制中'+str(frame_count))
                    else:
                        # > 50，停止，预测
                        last_action = self.calSimilarity(seq_feats)

                        # 播放声音
                        music_file = './voice/{}.wav'.format(last_action)
                        self.backPlay(music_file)
                        
                        # 重置
                        # record_status = False
                        frame_count = 0
                        seq_feats = []

                        record_status = True
                        triger_time = time.time()
                        print('start')
                else:
                    # 黄色3秒准备
                    cv2.circle(frame,(50,50),20,(0,255,255),-1)
            else:
                # 红色，等待
                cv2.circle(frame,(50,50),20,(0,0,255),-1)
            # 显示
            for y,x,score in results:
                x = int(x * frame_w)
                y = int(y * frame_h)
                cv2.circle(frame,(x,y),10,(0,255,0),-1)


            text = 'Pred: ' + last_action
            cv2.putText(frame,text,(50,150),cv2.FONT_ITALIC,1,(0,255,0),2)

            now = time.time()
            fps_time = now - start_time
            start_time = now

            fps_txt =round( 1/fps_time,2)
            cv2.putText(frame, str(fps_txt), (50,200), cv2.FONT_ITALIC, 1, (0,255,0),2)

            cv2.imshow('demo',frame)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  
                # 开始录制
                record_status = True
                triger_time = time.time()
                print('start')

                # 播放声音
                music_file = './voice/succ.wav'
                self.backPlay(music_file)
                pass
            elif pressedKey == ord("q"):  
                break
        

video = VideoFeat()  
video.realTimeVideo()