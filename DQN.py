import sys
# from environment import Env 이걸 만들어야해 환경 ^^ 재환아 지원아 수고해
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow
from enviornment import Env, GraphicDisplay
import gym

EPISODES = 300 # 판수 ㅋ


class DQNAgent:
    def __init__(self, state_size, action_size):  # 상태와 액션의 크기 받기
        self.render = False  # 이 친구는 바로 바로 바로 바로 그래픽을 보이게 말지 결정하는 친구
        self.load_model = False  # 이친구는 기존에 가중치를 가져올것인가를 결정하는 친

        self.discount_factor = 0.9  # 감가율 일단 0.9로 설정함

        self.epsilon = 1.0  # e-탐욕 정책사용, 탐험률 일단 1.0 설정
        self.epsilon_min = 0.01  # 탐험률의 최소 값 정의
        self.epsilon_decay = 0.999  # 탐험률의 감소값 일단 0.999로 설정
        self.train_start = 1000  # 학습을 시작하는 판수
        self.learning_rate = 0.001  # 인공신경망에 사용되는 학습

        self.state_size = state_size  # 인공지능망의 인풋 개수, 상태의 갯수랍니
        self.action_size = action_size  # 인공신경망에 아웃풋 개수! TODO 액션리스트중에서 할 수 없는것 가리

        self.batch_size = 64  # 한번에 학습을 위해 가져올 샘플 수, 조정 필

        self.memory = deque(maxlen=2000)  # 메모리 최대크기 2000으로 설정함

        self.model = self.build_agent_model()  # 행동 결정 인공신경망
        self.target_model = self.build_agent_model()  # 탈겟 인공신경망
        self.model_enemy = self.build_enemy_model()  # 상대 행동 예측 인공신경망
        self.target_model_enemy = self.build_enemy_model()  # 상대 행동 예측 탈겟 인공신경망

        self.update_target_model()  # 가중치 동기화 함수
        self.update_target_model_enemy()  # 가중치 동기화 합수

    #       if self.load_model(): TODO 가중치 불러오긴가 뭔가 그거 해놓기 주소 만들
    #          self.model.load_weights("./save_model/cartpole_dqn_trained.h5") ## 일단 책에 중요해 보여서 썼는데, 뭔 코든지 해석좀

    def build_agent_model(self):  # 행동을 결정하는 인공신경망
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size + 1, activation='relu',
                        kernel_initializer='he_uniform'))  # 여기서는 상대의 예상행동도 인수로 받기때문에 state+1
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_enemy_model(self):  # 에네미 행동을 추측하는 인공신경
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):  # 가중치 동기화
        self.target_model.set_weights(self.model.get_weights())

    def update_target_model_enemy(self):  # 가중치 동기화
        self.target_model_enemy.set_weights(self.model_enemy.get_weights())

    def get_action(self, state, enemy_action):  # 인공신경망을 이용해서 상태와 상대 예상행동을 받고 행동을 리턴
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 탐험률
        else:
            state.append(enemy_action)  # 상태에 상대방 예상 행동도 추가해서 인공신경망에 넣기
            q_value = self.model.predict(state)
            return list(q_value[0])

    def get_enemy_action(self, enemy_state):  # 에네미의 예상행동을 반환하는 함수, 탐험 없고 상대 행동받는것 없음
        q_value = self.model.predict(enemy_state)
        return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):  # 학습샘플을 메모리에 저장하는 함수
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):  # 학습시키기
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay  # 입실론 조정치를 조금씩 줄여감

        mini_batch = random.sample(self.memory, self.batch_size)  # 메모리에서 랜덤으로 배치 사이즈많큼 샘플을 뽑아옴

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
                target[i][actions[i]] = rewards[i]  # 벨만기대방정식을 이용해서 인공신경망 업데이
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))
        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def train_model_enemy(self, memory):  # 적의 메모리를 인수로 받아 학습
        mini_batch = random.sample(memory, self.batch_size)  # 상대 메모리에서 샘플 추출

        states = np.zeros((self.batch_size, self.state_size))  # 세로열이 batch_size, state_size 가로열 개수인 0으로 도배된 배열 생성
        next_states = np.zeros((self.batch_size, self.state_size))

        actions, rewards, dones = [], [], []
        for i in range(self.batch_size):  # 이건바뀔필요가 없어 그냥 상수
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])  # 모든 변수에 샘플을 정리
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model_enemy.predict(states)
        target_val = self.target_model_enemy.predict(next_states)
        for i in range(self.batch_size):
            if done[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))
        self.model_enemy.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":

    env = Env  # 환경 가저오기!
    state_size = env.state_space  # TODO 환경에 상태의 크기를 가져오는 함수가 필요함
    action_size = env.action_space  # TODO 환경에 액션의 크기를 가져오는 함수가 필여

    agent_1 = DQNAgent(state_size, action_size)
    agent_2 = DQNAgent(state_size, action_size)  # 이제부터 환경의 함수가 필요한데.,,, 내일 말조 ㅁ해보고 하자

    scores_1, episodes_1, scores_2, episodes_2 = [], [], [], []

    for e in range(EPISODES): # g한판이라고 피면 ㅇㅇ 치면 ㅇㅇ 치면 ㅇㅇ 치면 ㅇㅇ 치면 done가 겜 끝났나 아닌가 ㅇㅋ?
        done = False
        score_1 = 0
        score_2 = 0

        state_1 = env.reset()  # TODO 스테이트 초기화를 뭔 매턴마다해 바봏여
        state_1 = np.reshape(state_1, [1, state_size])  # TODO 스테이트 촉화를 뭐이렇게 많이

        state_2 = None  # TODO stage의 상대 상태와 자신 상태 바꿔주기

        while not done:
            if render:
                GraphicDisplay.show()  # Todo render 설정해놓으면 볼 수 있게 하는 함수야

            if env.getFirstAgent() == 1:
                spit_list_1 = [0, 0, 0]
                while True:
                    q_value_1 = self.model.predict(state_1)
                    q_value_simple_1 = q_value_1[:4]
                    if state_1[5+np.argmin(q_value_1[0])] - 1 < 0:
                        del q_value_simple_1[np.argmin(q_value_1[0])]
                    else:
                        spit_list_1[0] = np.argmin(q_value_1[0])
                        break
                while True:
                    state_1_strange = state_1[5+np.argmin(q_value_1[0])] -1
                    q_value_2 = self.model.predict(state_1_strange)
                    q_value_simple_2 = q_value_2[:4]
                    if state_1_strange[5+np.argmin(q_value_2[0])] - 1 < 0:
                        del q_value_simple_2[np.argmin(q_value_2[0])]
                    else:
                        spit_list_1[1] = np.argmin(q_value_2[0])
                        break
                while True:
                    state_1_strange_strange = state_1_strange[5+np.argmin(q_value_2[0])] - 1
                    q_value_3 = self.model.predict(state_1_strange_strange)
                    q_value_simple_3 = q_value_3[:4]
                    if state_1_strange_strange[5+np.argmin(q_value_3[0])] -1 < 0:
                        del q_value_simple_3[np.argmin(q_value_2[0])]
                    else:
                        spit_list_1[2] = np.argmin(q_value_3[0])
                        break


                action_1 = agent_1.model(state_1, agent_1.get_enemy_action(state_2))  # 상대의 액션을 넣어서 한
                next_state_1, reward_1, done = env.step(action_1, spit_list_1)  # TODO env에 한 액션을 받고 한 턴을 플레이 하는 함수르 만들기
                action_2 = agent_2.get_action(state_2, agent_2.get_enemy_action(state_1))
                next_state_2, reward_2, done = env.step(action_2)  # 한턴 진행한거야 먼저한친구가 1이면 이렇게
            if env.getFirstAgent() == 2:
                spit_list_2 = [0, 0, 0]
                while True:
                    q_value_4 = self.model.predict(state_2)
                    q_value_simple_4 = q_value_4[:4]
                    if state_2[5 + np.argmin(q_value_4[0])] - 1 < 0:
                        del q_value_simple_4[np.argmin(q_value_4[0])]
                    else:
                        spit_list_2[0] = np.argmin(q_value_4[0])
                        break
                while True:
                    state_2_strange = state_2[5 + np.argmin(q_value_4[0])] - 1
                    q_value_5 = self.model.predict(state_2_strange)
                    q_value_simple_5 = q_value_5[:4]
                    if state_2_strange[5 + np.argmin(q_value_5[0])] - 1 < 0:
                        del q_value_simple_5[np.argmin(q_value_5[0])]
                    else:
                        spit_list_2[1] = np.argmin(q_value_5[0])
                        break
                while True:
                    state_2_strange_strange = state_2_strange[5 + np.argmin(q_value_6[0])] - 1
                    q_value_6 = self.model.predict(state_2_strange_strange)
                    q_value_simple_6 = q_value_6[:4]
                    if state_2_strange_strange[5 + np.argmin(q_value_6[0])] - 1 < 0:
                        del q_value_simple_6[np.argmin(q_value_6[0])]
                    else:
                        spit_list_2[2] = np.argmin(q_value_6[0])
                        break
                action_2 = agent_2.get_action(state_2, agent_2.get_enemy_action(state_1))
                next_state_2, reward_2, done = env.step(action_2)  # 한턴 진행한거야 먼저한친구가 2이면 이렇게
                action_1 = agent_1.get_action(state_1, agent_1.get_enemy_action(state_2))
                next_state_1, reward_1, done = env.step(action_1)  # TODO env에 한 액션을 받고 한 턴을 플레이 하는 함수르 만들기

            next_state_1 = np.reshape(next_state_1, [1, state_size])
            next_state_2 = np.reshape(next_state_2, [1, state_size])

            agent_1.append_sample(state_1, action_1, reward_1, next_state_1, done)
            agent_2.append_sample(state_2, action_2, reward_2, next_state_2, done)

            if len(agent_1.memory) >= agent_1.train_start:
                agent_1.train_model()
                agent_1.train_model_enemy(agent_2.memory)
                # if len(agent_2.memory) >= agent.train_start:    메모리 개수는 똑같아서 그냥 한 함수에 둘다 넣었어.
                agent_2.train_model()
                agent_2.train_model_enemy(agent_1.memory)

            score_1 += reward_1
            score_2 += reward_2

            state_1 = next_state_1
            state_2 = next_state_2

            if done:
                agent_1.update_target_model() #가중치 통일
                agent_1.update_target_model_enemy()# 가중치 통일
                agent_2.update_target_model()# 가중치 통
                agent_2.update_target_model_enemy() # 가중치 통일

                scores_2.append(score_2)
                episodes_2.append(e)  # <-- 이 e가 뭐하는걸까 모르겟음
                pylab.plot(episodes_2, scores_2, 'b')

                scores_1.append(score_1)
                episodes_1.append(e)  # <-- 이 e가 뭐하는걸까 모르겟음
                pylab.plot(episodes_1, scores_1, 'b')

                # pylab.savefig("")    <- 그래프 저장 경로 사실 딱히 필요없음 자소서 쓸때 필요하려나
                if np.mean(scores_2[-min(10, len(scores_2)):]) > 100:  # 이전 에피소드 점수 평균이 100넘으면 탈출 (학습)
                    # agent.model.save_weights() <-- 이괄호 안에는 학습 데으터를 넣을 경로가 필요한데 아직 못정함
                    sys.exit()
                    # 아래있는거 그래프 + 가중치 동기화
                    # 그래프 + 가중치 기화
