# -*- coding:utf-8 -*-
"""
作者：知行合一
日期：2019年 10月 7日 9:50
文件名：action.py
地点：changsha
"""
import os.path

import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite
import glob
import tqdm
import os
# 导入DTW算法包
from fastdtw import fastdtw



"""
骨骼点动态动作识别
1、计算所有训练集视频片段特征，并存入文件
    计算每个视频片段的特征
        计算每一帧画面的特征
            特征由两点之间形成的向量构成的夹角表示
                夹角由像素坐标构成
    
2、计算实时视频流，截取一段序列，计算特征
3、计算实时视频流特征与训练集特征的DTW距离
4、排序，筛选，预测动作
"""


class VideoFeature:
    """
    获取人体Pose关键点
    获取画面特征
    获取视频片段特征
    """

    def __init__(self):
        # 加载movenet关键点检测模型
        self.interpreter = tflite.Interpreter(
            model_path="./weights/hub/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def getFramePose(self, image):
        """
        获取关键点
        """
        # 转为RGB
        img_cvt = cv2.resize(image, (192, 192))
        img_cvt = cv2.cvtColor(img_cvt, cv2.COLOR_BGR2RGB)
        img_input = img_cvt.reshape(1, 192, 192, 3)

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

    def getVectorAngle(self, v1, v2):
        """
        返回向量之间的夹角
        """
        # 判断向量是否一致
        if np.array_equal(v1, v2):
            return 0
        dot_product = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)

        return np.arccos(dot_product / norm)

    def getFrameFeat(self,frame):
        """
        获取画面特征
        """
        # 获取画面关键点
        keypoints_with_scores = self.getFramePose(frame)
        # 重新组织为x,y的列表
        keypoints_list = [[landmark[1],landmark[0]] for landmark in keypoints_with_scores]

        # 根据关键点构造向量
        # 关键点连接关系
        conns = [(0,1),(0,2),(1,3),(3,5),(2,4)]

        # 转为numpy数组
        keypoints_list = np.asarray(keypoints_list)

        # 向量列表
        # keypoints_list[1] - keypoints_list[0]
        vector_list = list(map(
            lambda conn: keypoints_list[conn[1]] - keypoints_list[conn[0]],
            conns
        ))

    # 计算任意两个向量之间的夹角
        angle_list = []
        for vector_a in vector_list:
            for vector_b in vector_list:
                angle = self.getVectorAngle(vector_a,vector_b)
                angle_list.append(angle)
        return angle_list,keypoints_list

    def getVideoFeat(self,videoFile):
        """
        获取视频片段特征
        """
        cap = cv2.VideoCapture(videoFile)
        video_feat = []
        while True:
            ret,frame = cap.read()
            if frame is None:
                break

            # 获取每一帧画面特征
            frame_feat,_ = self.getFrameFeat(frame)
            video_feat.append(frame_feat)
        cap.release()
        return video_feat

    def getTrainingFeats(self):
        """
        计算所有训练集视频片段特征
        如果有trainingDate.npz文件，则返回其中内容，如果没有再临时计算
        """
        saveFileNme = './data/trainingDate.npz'

        # 检查是否存在文件
        if os.path.exists(saveFileNme):
            with open(saveFileNme,'rb') as f:
                return np.load(f,allow_pickle='TRUE')
        # 没有该文件，需要重新生成
        filename = r'.\data\action_train\*\*.mp4'
        file_list = glob.glob(filename)
        training_feat = []

        # 遍历所有视频片段
        for file in tqdm.tqdm(file_list,desc='训练集处理中'):
            # 处理单个片段
            video_feat = self.getVideoFeat(file)
            action_name = file.split('\\')[3]
            training_feat.append([action_name,video_feat])

        # 转为numpy数组
        training_feat = np.array(training_feat,dtype=object)
        # 写入文件
        with open(saveFileNme,'wb') as f:
            np.save(f,training_feat)
        # 返回该数据
        return training_feat










class Pose_recognition:
    def __init__(self):
        # 实例化
        self.video_feat = VideoFeature()
        # 获取训练集特征
        self.training_feat = self.video_feat.getTrainingFeats()

        # 指定前batch_size
        self.batch_size = 4
        self.threshold = 0.5
    # 计算相似度
    def calSimilarity(self,seq_feats):
        """
        计算序列特征seq_feats 与训练集特征集合的距离，并返回预测动作

        """
        # 遍历计算序列与训练集特征
        dist_list = []
        for action_name,video_feat in self.training_feat:
            # 计算DTW距离
            distance,path = fastdtw(seq_feats,video_feat)
            dist_list.append([action_name,distance])


        # 转为numpy数组
        dist_list = np.array(dist_list,dtype=object)
        # 排序(距离由低到高),筛选前batch_size 个列表
        dist_list = dist_list[dist_list[:,1].argsort()][:self.batch_size]

        # 获取排名第一的动作名称
        first_key = dist_list[0][0]

        # 检查该动作再前batch_size中重复次数
        max_num = np.count_nonzero(dist_list[:,0] == first_key)
        # 判断该次数是否大于阈值
        if max_num /self.batch_size >=  self.threshold:
            return first_key
        else:
            return 'unknow'



    def recognize(self):

        cap = cv2.VideoCapture(0)
        # 视频宽度和高度
        frme_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frme_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 是否录制
        recrod_status = False

        # 记录录制帧数
        frame_count = 0

        # 序列特征
        squen_feat = []

        # 触发时间
        triger_time = time.time()

        while True:

            ret,frame = cap.read()
            # 获取每一帧画面特征
            angle_list, keypoints_list = self.video_feat.getFrameFeat(frame)

            for x,y in keypoints_list:
                x = int(x * frme_w)
                y = int(y * frme_h)
                # 画原点
                cv2.circle(frame,(x,y),10,(0,255,0),-1)

            if recrod_status:
                if time.time() - triger_time > 3:
                    # 开始录制为绿色绘制
                    cv2.circle(frame, (70, 50), 20, (0, 255, 0), -1)
                    # 开始录制
                    if frame_count < 50:
                        # 继续录制
                        # 获取每一帧画面特征
                        # frame_feat = self.video_feat.getFrameFeat(frame)
                        squen_feat.append(angle_list)
                        frame_count += 1
                    else:
                        # 停止录制并且开始识别
                        # 使用DTW算法计算序列特征与训练集的距离
                        print('开始识别')
                        action_name = self.calSimilarity(squen_feat)
                        print(action_name)

                        # 初始化
                        frame_count = 0
                        # recrod_status = False
                        squen_feat = []
                        triger_time = time.time()

                else:
                    # 启动，等待录制绘制黄色
                    cv2.circle(frame, (70, 50), 20, (0, 255, 255), -1)

            else:
                # 等待状态绘制红色
                cv2.circle(frame, (70, 50), 20, (0, 0, 255), -1)

            cv2.imshow('action',frame)

            press_key = cv2.waitKey(1) & 0xFF
            if press_key == ord('q'):
                break
            elif press_key == ord('r'):
                # 启动识别
                recrod_status = True
                triger_time = time.time()
                print('开始录制')



        cap.release()
        cv2.destroyAllWindows()


pose = Pose_recognition()
pose.recognize()
