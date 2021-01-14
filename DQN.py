import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQNAgent:
    def __init__(self, state_size, action_size): # 상태와 액션의 크기 받기
        self.discount_factor = 0.9 #감가율 일단 0.9로 설정함

        self.epsilon = 1.0 # e-탐욕 정책사용, 탐험률 일단 1.0 설정
        self.epsilon_min = 0.01 #탐험률의 최소 값 정의
        self.epsilon_decay = 0.999 # 탐험률의 감소값 일단 0.999로 설정

        self.state_size = state_size
        self.action_size = action_size


        self.memory = deque(maxlen=2000)#메모리 최대크기 2000으로 설정함

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()

 #       if self.load_model():
#          self.model.load_weights("./save_model/cartpole_dqn_trained.h5") ## 일단 책에 중요해 보여서 썼는데, 뭔 코든지 해석좀

    def build_model(self): # 인공신경망 만들
        model = Sequential()





