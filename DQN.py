import sys
#from environment import Env 이걸 만들어야해 환경 ^^ 재환아 지원아 수고해
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 300

class DQNAgent:
    def __init__(self, state_size, action_size): # 상태와 액션의 크기 받기
        self.discount_factor = 0.9 #감가율 일단 0.9로 설정함

        self.epsilon = 1.0 # e-탐욕 정책사용, 탐험률 일단 1.0 설정
        self.epsilon_min = 0.01 #탐험률의 최소 값 정의
        self.epsilon_decay = 0.999 # 탐험률의 감소값 일단 0.999로 설정

        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = 64 # 한번에 학습을 위해 가져올 샘플 수, 조정 필

        self.memory = deque(maxlen=2000)#메모리 최대크기 2000으로 설정함

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()


 #       if self.load_model():
#          self.model.load_weights("./save_model/cartpole_dqn_trained.h5") ## 일단 책에 중요해 보여서 썼는데, 뭔 코든지 해석좀

    def build_model(self): # 인공신경망 만들어야 하는데, 상의가 필요함
        model = Sequential()
        #pass
        return model



    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else :
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done): # 학습샘플을 메모리에 저장하는 함수
        self.memory.append((state,action, reward, next_state, done))

    def train_model(self): # 학습시키기
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay요

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))

        actions, rewards, dones = [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])


        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if done[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))
        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose= 0)

if __name__ == "__main__":


    ##agent = DQNAgent(state_size, action_size) 이제부터 환경의 함수가 필요한데.,,, 내일 말조 ㅁ해보고 하자


    for e in range(EPISODES):
        done = False







