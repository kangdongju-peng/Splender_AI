import sys
# from environment import Env 이걸 만들어야해 환경 ^^ 재환아 지원아 수고해
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from enviornment import Env, GraphicDisplay
import gym

EPISODES = 300


class DQNAgent:
    def __init__(self, state_size, action_size):  # 상태와 액션의 크기 받기
        self.discount_factor = 0.9  # 감가율 일단 0.9로 설정함

        self.epsilon = 1.0  # e-탐욕 정책사용, 탐험률 일단 1.0 설정
        self.epsilon_min = 0.01  # 탐험률의 최소 값 정의
        self.epsilon_decay = 0.999  # 탐험률의 감소값 일단 0.999로 설정
        self.train_start = 1000
        self.learning_rate = 0.001

        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = 64  # 한번에 학습을 위해 가져올 샘플 수, 조정 필

        self.memory = deque(maxlen=2000)  # 메모리 최대크기 2000으로 설정함

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.enemy_action
        self.enemy_state

        self.update_target_model()

    #       if self.load_model():
    #          self.model.load_weights("./save_model/cartpole_dqn_trained.h5") ## 일단 책에 중요해 보여서 썼는데, 뭔 코든지 해석좀

    def build_agent_model(self):  # 인공신경망 만들어야 하는데, 상의가 필요함
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # pass
        return model

    def update_target_agent_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def get_enemy_action(self, enemy_state):
        q_value = self.model.predict(enemy_state)
        return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):  # 학습샘플을 메모리에 저장하는 함수
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):  # 학습시키기
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))  # 세로열이 batch_size, state_size 가로열 개수인 0으로 도배된 배열 생성
        next_states = np.zeros((self.batch_size, self.state_size))

        actions, rewards, dones = [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])  # 모든 변수에 샘플을 정리
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
        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":

    env = Env  # 환경 가저오기!
    state_size = env.state_space  #TODO 환경에 상태의 크기를 가져오는 함수가 필요함
    action_size = env.action_space  #TODO 환경에 액션의 크기를 가져오는 함수가 필여
    agent = DQNAgent(state_size, action_size)  # 이제부터 환경의 함수가 필요한데.,,, 내일 말조 ㅁ해보고 하자

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        if env.getTurn():
            state = env.reset()
            state = env.startset()
        else:
            enemy_state = env.reset()  #TODO <-- 상태 리셋하는 함수 env에 넣어야함
            enemy_state = env.start_set()  #TODO <-- 시작할때 상태를 나타내는거

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)  #TODO env에 한 액션을 받고 한 턴을 플레이 하는 함수르 만들기
            next_state = np.reshape(next_state, [1, state_size])
            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            score += reward
            state = next_state
            if done:
                agent.update_target_model()
                scores.append(score)
                episodes.append(e)  # <-- 이 e가 뭐하는걸까 모르겟음
                pylab.plot(episodes, scores, 'b')
                # pylab.savefig("")    <- 그래프 저장 경로 사실 딱히 필요없음 자소서 쓸때 필요하려나
                if np.mean(scores[-min(10, len(scores)):]) > 100:  # 이전 에피소드 점수 평균이 100넘으면 탈출 (학습)
                    # agent.model.save_weights() <-- 이괄호 안에는 학습 데으터를 넣을 경로가 필요한데 아직 못정함
                    sys.exit()
