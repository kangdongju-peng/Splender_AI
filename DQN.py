import sys
# from environment import Env 이걸 만들어야해 환경 ^^ 재환아 지원아 수고해
from functools import reduce

import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow
import torch
from enviornment import Env
import gym

EPISODES = 10000  # 판수 ㅋ


class DQNAgent:
    def __init__(self, state_size, action_size):  # 상태와 액션의 크기 받기

        self.render = False  # 이 친구는 바로 바로 바로 바로 그래픽을 보이게 말지 결정하는 친구
        self.load_model = False  # 이친구는 기존에 가중치를 가져올것인가를 결정하는 친

        self.discount_factor = 0.9  # 감가율 일단 0.9로 설정함

        self.epsilon = 1.0  # e-탐욕 정책사용, 탐험률 일단 1.0 설정
        self.epsilon_min = 0.01  # 탐험률의 최소 값 정의
        self.epsilon_decay = 0.999  # 탐험률의 감소값 일단 0.999로 설정
        self.train_start = 2000  # 학습을 시작하는 메모리수
        self.learning_rate = 0.001  # 인공신경망에 사용되는 학습

        self.state_size = state_size  # 인공지능망의 인풋 개수, 상태의 갯수랍니
        self.action_size = action_size  # 인공신경망에 아웃풋 개수! TODO 액션리스트중에서 할 수 없는것 가리

        self.batch_size = 500  # 한번에 학습을 위해 가져올 샘플 수, 조정 필

        self.memory = deque(maxlen=5000)  # 메모리 최대크기 2000으로 설정함

        self.model = self.build_agent_model()  # 행동 결정 인공신경망
        self.target_model = self.build_agent_model()  # 탈겟 인공신경망

        self.update_target_model()  # 가중치 동기화 함수

        if self.load_model:
            self.model.load_weights("model.h5") ## 일단 책에 중요해 보여서 썼는데, 뭔 코든지 해석좀

    def build_agent_model(self):  # 행동을 결정하는 인공신경망
        leaky_relu = tensorflow.nn.leaky_relu
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size + 1, activation=leaky_relu,
                        kernel_initializer='he_uniform'))  # 여기서는 상대의 예상행동도 인수로 받기때문에 state+1
        model.add(Dense(24, activation=leaky_relu, kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):  # 가중치 동기화
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, enemy_action):  # 인공신경망을 이용해서 상태와 상대 예상행동을 받고 행동을 리턴
        if np.random.rand() <= self.epsilon:
            return_list = []
            for n in range(17):
                return_list.append(random.random())
            for n in range(15):
                return_list.append(random.random()/2)
            return return_list  # 탐험률
        else:
            del state[self.state_size - 1]
            state.append(enemy_action)  # 상태에 상대방 예상 행동도 추가해서 인공신경망에 넣기
            state = np.reshape(state, [1, state_size + 1])
            q_value = self.model.predict(state)
            q_value = list(q_value[0])

            return q_value

    def get_enemy_action(self, enemy_state):  # 에네미의 예상행동을 반환하는 함수, 탐험 없고 상대 행동받는것 없음
        q_value = self.model.predict(enemy_state)
        return list(q_value[0]).index(max(q_value[0]))

    def append_sample(self, state, action, reward, next_state, done):  # 학습샘플을 메모리에 저장하는 함수
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):  # 학습시키기
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay  # 입실론 조정치를 조금씩 줄여감

        mini_batch = random.sample(self.memory, self.batch_size)  # 메모리에서 랜덤으로 배치 사이즈많큼 샘플을 뽑아옴

        states = np.zeros((self.batch_size, self.state_size + 1))  # 세로열이 batch_size, state_size 가로열 개수인 0으로 도배된 배열 생성
        next_states = np.zeros((self.batch_size, self.state_size + 1))

        actions, rewards, dones = [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])  # 모든 변수에 샘플을 정리
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3] + [1]
            dones.append(mini_batch[i][4])

        target = list(self.model.predict(states))
        target_val = list(self.target_model.predict(next_states))
        for i in range(self.batch_size):
            _target = list(target[i])
            if dones[i]:
                _target[actions[i]] = rewards[i]  # 벨만기대방정식을 이용해서 인공신경망 업데이

            else:
                _target[int(actions[i])] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))
            target[i] = _target
            self.model.fit(np.reshape(states[i], [1, len(states[i])]), np.reshape(target[i], [1, len(target[i])]),
                           batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":

    env = Env()  # 환경 가저오기!
    state_size = 174  # TODO 환경에 상태의 크기를 가져오는 함수가 필요함
    action_size = 36  # TODO 환경에 액션의 크기를 가져오는 함수가 필여

    agent_1 = DQNAgent(state_size, action_size)

    scores_1, episodes_1, scores_2, episodes_2 = [], [], [], []

    for e in range(EPISODES):  # g한판이라고 피면 ㅇㅇ 치면 ㅇㅇ 치면 ㅇㅇ 치면 ㅇㅇ 치면 done가 겜 끝났나 아닌가 ㅇㅋ?
        done = False
        score_1 = 0
        score_2 = 0

        env.setting()
        state_1 = env.state_return(0) + [1]  # TODO 스테이트 초기화를 뭔 매턴마다해 바봏여
        state_1 = np.reshape(state_1, [1, state_size + 1])  # TODO 스테이트 촉화를 뭐이렇게 많이

        while not done:
            # if render:
            #    GraphicDisplay.show()  # Todo render 설정해놓으면 볼 수 있게 하는 함수야

            for n in range(2):
                spit_list_1 = [-1, -1, -1]
                state_1 = env.state_return(env.turn) + [1]

                action_1 = agent_1.get_action(state_1, agent_1.get_enemy_action(
                    np.reshape(env.state_return((lambda x: 0 if x == 1 else 1)(n)) + [1],
                               [1, state_size + 1])))  # 상대의 액션을 넣어서 한
                _state, reward_1, done = env.step(q_val=list(action_1))

                if done:
                    break
                _state = np.reshape(_state + [1], [1, state_size + 1])

                token_count = sum(env.my_token[(lambda x: 1 if x == 0 else 0)(env.turn)])
                for i in range(3):
                    q_value_1 = list(agent_1.model.predict(_state)[0])
                    q_value_simple_1 = q_value_1[:5] + q_value_1[35]
                    j = 0
                    while token_count > 10:
                        # print(int(np.argmin(q_value_simple_1)))
                        if _state[0, 6 + int(np.argmin(q_value_simple_1))] <= 0:
                            q_value_simple_1[int(np.argmin(q_value_simple_1))] = max(q_value_simple_1) + 1
                        else:
                            spit_list_1[i] = int(np.argmin(q_value_simple_1))
                            # print(spit_list_1)
                            _state[0, 6 + int(np.argmin(q_value_simple_1))] -= 1
                            token_count -= 1
                            break
                if sum(env.my_token[(lambda x: 1 if x == 0 else 0)(env.turn)]) > 10:
                    env.spit(spit_list_1)

                reward_1 = env.reward
                print(env.ac_color + str(reward_1) + "\033[0m")
                score_1 += reward_1
                action_1 = env.acted_action
                agent_1.append_sample(state_1, action_1, reward_1, env.state_return(n), done)
            if len(agent_1.memory) >= agent_1.train_start:
                agent_1.train_model()

            # env.state_print(env.state_return(1))
            # print(str(env.my_score))
            if done:
                print("\033[93m" + "#" + str(e) + " done " + "\033[0m",
                      "\033[95m" + str(env.my_score) + "\033[0m", len(agent_1.memory))
                env.state_print(env.state_return(0))
                agent_1.update_target_model()  # 가중치 통일

                scores_2.append(score_2)
                episodes_2.append(e)  # <-- 이 e가 뭐하는걸까 모르겟음
                pylab.plot(episodes_2, scores_2, 'b')

                scores_1.append(score_1)
                episodes_1.append(e)  # <-- 이 e가 뭐하는걸까 모르겟음
                pylab.plot(episodes_1, scores_1, 'b')

                if e % 5 == 0:
                    model_json = agent_1.model.to_json()
                    with open("model.json", "w") as json_file:
                        json_file.write(model_json)
                    agent_1.model.save_weights("model.h5")
                    print("saved model")

                if np.mean(scores_2[-min(10, len(scores_2)):]) > 14:  # 이전 에피소드 점수 평균이 100넘으면 탈출 (학습)
                    sys.exit()
                    # 아래있는거 그래프 + 가중치 동기화
                    # 그래프 + 가중치 기화
                break
                # pylab.savefig("")    <- 그래프 저장 경로 사실 딱히 필요없음 자소서 쓸때 필요하려나
