from random import *
from functools import reduce
from typing import List, Any
import math
import numpy as np


# 이곳에 있는 모든 토큰은 white blue green red black gold 순으로
# 0 1 2 3 4 5 의 번호를 가짐을 알립니다

class Env:
    cards_now: List[List[Any]]
    cards_game: List[List[Any]]

    def __init__(self):
        self.original_score = []
        self.dkssud2 = 0
        self.dkssud = 0

        self.ac_color = ''
        self.acted_action = 0

        self.gamma = 0.95
        self.reward = 0
        self.used_tokens = 0

        self.turn = 0
        self.lords = ('340004',
                      '344000',
                      '333300',
                      '300044',
                      '300333',
                      '304400',
                      '330033',
                      '303330',
                      '300440',
                      '333003',)  # 귀족 종류
        self.cards = (('k011110',
                       'k012110',
                       'k022010',
                       'k000131',
                       'k000210',
                       'k020200',
                       'k000300',
                       'k104000',
                       'u010111',
                       'u010121',
                       'u010220',
                       'u001310',
                       'u010002',
                       'u000202',
                       'u000003',
                       'u100040',
                       'w001111',
                       'w001211',
                       'w002201',
                       'w031001',
                       'w000021',
                       'w002002',
                       'w003000',
                       'w100400',
                       'g011011',
                       'g011012',
                       'g001022',
                       'g013100',
                       'g021000',
                       'g002020',
                       'g000030',
                       'g100004',
                       'r011101',
                       'r021101',
                       'r021101',
                       'r020102',
                       'r010013',
                       'r002100',
                       'r020020',
                       'r030000',),
                      ('k132200',
                                     'k130302',
                                     'k201420',
                                     'k200530',
                                     'k205000',
                                     'k300006',
                                     'u102230',
                                     'u102303',
                                     'u253000',
                                     'u220014',
                                     'u205000',
                                     'u306000',
                                     'w100322',
                                     'w123030',
                                     'w200142',
                                     'w200053',
                                     'w200050',
                                     'w360000',
                                     'g130230',
                                     'g123002',
                                     'g242001',
                                     'g205300',
                                     'g200500',
                                     'g300600',
                                     'r120023',
                                     'r103023',
                                     'r214200',
                                     'r230005',
                                     'r200005',
                                     'r300060',),
                      ('k333530',
                                                   'k400363',
                                                   'k400070',
                                                   'k500073',
                                                   'u330335',
                                                   'u470000',
                                                   'u463003',
                                                   'u573000',
                                                   'w303353',
                                                   'w400007',
                                                   'w430036',
                                                   'w530007',
                                                   'g353033',
                                                   'g407000',
                                                   'g436300',
                                                   'g507300',
                                                   'r335303',
                                                   'r400700',
                                                   'r400700',
                                                   'r500730',),)  # 카드 종류
        self.setting()
        self.state_space = 126
        self.Token = {'w': 0,
                      'b': 1,
                      'g': 2,
                      'r': 3,
                      'k': 4}
        self.done = False

        self.turn = 0  # 턴 0번플레이어, 1번플레이어로 나뉨

    def setting(self):
        self.Token = {'w': 0,
                      'u': 1,
                      'g': 2,
                      'r': 3,
                      'k': 4}
        self.done = False

        self.turn = 0  # 턴 0번플레이어, 1번플레이어로 나뉨
        self.cards_game = [[], [], []]  # 순서 없이 섞인 카드들

        self.cards_now = [[], [], []]  # 현제 나와있는 1 2 3티어 카드
        self.lord_now = None  # 본 판에 나와있는 귀족들
        self.tokens = [4, 4, 4, 4, 4, 5]

        self.my_score = [0, 0]
        self.my_token = [[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]]  # 내 카드 #상대카
        self.my_card = [[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]  # 내 카드 # 상대 카드
        self.my_kept_card = [[], []]  # 내가 찜한 카드
        self.my_lord = [[], []]

        self.action_size = 30

        for i in range(0, 3):
            self.cards_game[i] = list(self.cards[i])
        for i in range(0, 3):
            shuffle(self.cards_game[i])
        for i in range(0, 3):
            self.cards_game[i] = list(self.cards[i])
            shuffle(self.cards_game[i])
            self.cards_now[i] = self.cards_game[i][:4]
            del self.cards_game[i][:4]
        self.lord_now = sample(self.lords, 3)
        return

    # 상태를 초기화하는 함수
    def state_return(self, turn):
        ind = (lambda x: 1 if x == 0 else 0)(turn)
        turn, ind = ind, turn
        state = self.tokens + self.my_token[turn] + \
                self.my_token[ind] + self.my_card[turn] + self.my_card[ind]
        for cards in self.cards_now:
            for card in cards:
                state.append(self.Token[card[0]])
                for i in range(1, 7):
                    state.append(int(card[i]) + 1)

        full3 = lambda t: self.my_kept_card[t] if len(self.my_kept_card[t]) == 3 else self.my_kept_card[t] + [0] if len(
            self.my_kept_card[t]) == 2 else self.my_kept_card[t] + [0, 0] if len(self.my_kept_card[t]) == 1 \
            else [0, 0, 0]

        for kp_card in full3(turn):
            if kp_card == 0:
                state += [0, 0, 0, 0, 0, 0, 0]
                continue
            state.append(self.Token[kp_card[0]] + 1)
            for i in range(1, 7):
                state.append(int(kp_card[i]))
        for kp_card in full3(ind):
            if kp_card == 0:
                state += [0, 0, 0, 0, 0, 0, 0]
                continue
            state.append(self.Token[kp_card[0]] + 1)
            for i in range(1, 7):
                state.append(int(kp_card[i]))

        for lord in self.lord_now:
            for i in range(len(lord)):
                state.append(int(lord[i]))
        state += [self.my_score[turn], self.my_score[ind]]
        return state

    def state_print(self, state):
        print("\033[96m" + "tokens\t\t:" + str(state[:6]) + "\033[0m")
        print("\033[96m" + "my token\t:" + str(state[6:12]) + "\033[0m")
        print("\033[96m" + "ene token\t:" + str(state[12:18]) + "\033[0m")
        print("\033[96m" + "kp card\t:" + str(self.my_kept_card[0]) + str(self.my_kept_card[1]) + "\033[0m")
        print("\033[96m" + "card\t:" + str(self.my_card[0]) +
              "/" + str(self.my_card[1]) + "\033[0m")
        print("\033[92m" + "card_all:" + "\033[0m")
        print(str(self.cards_now[0]))
        print(str(self.cards_now[1]))
        print(str(self.cards_now[2]))
        # print("\033[96m" + "score\t\t:" + str(self.my_score) + "\033[0m")

    # 토큰을 가져오는 함수
    # collect(가져올 토큰:int, 가져올 토큰/필수아님, 가져올 토큰/필수아님)
    def collect(self, *to_collect):
        # print(to_collect)
        if len(to_collect) == 1:
            # check collectable
            if self.tokens[to_collect[0]] >= 4:
                self.tokens[to_collect[0]] -= 2
                self.my_token[self.turn][to_collect[0]] += 2
            else:
                return False
        elif len(to_collect) == 3:
            if len(list(filter(lambda x: self.tokens[x] > 0, [0, 1, 2, 3, 4]))) <= 2:
                for jem in filter(lambda x: self.tokens[x] > 0, [0, 1, 2, 3, 4]):
                    self.tokens[jem] -= 1
                    self.my_token[self.turn][jem] += 1
            else:
                for jem in to_collect:
                    if self.tokens[jem] <= 0:
                        return False
                for jem in to_collect:
                    self.tokens[jem] -= 1
                    self.my_token[self.turn][jem] += 1
        self.ac_color = "\033[33m"
        self.acted_action = 1
        return True

    # 킵하는 함수
    # ** card 는 [n, m]형태의 리스트로 n = 티어 m = 번째 / 5번째 = 뒷면
    def keep(self, card):
        card[0] -= 1
        card[1] -= 1
        if len(self.my_kept_card[self.turn]) >= 3:
            return False
        if self.cards_now[card[0]][card[1]] == 'w000000':
            return False
        # print(self.tokens[5])
        if self.tokens[5] > 0:
            self.tokens[5] -= 1
            self.my_token[self.turn][5] += 1
            # print(self.my_token)
        if card[1] == 4:
            self.my_kept_card[self.turn].append(self.cards_game[card[0]][0])
            del self.cards_game[card[0]][0]
        else:
            self.my_kept_card[self.turn].append(self.cards_now[card[0]][card[1]])
            if len(self.cards_game[card[0]]) <= 0:
                self.cards_now[card[0]][card[1]] = 'w0000000'
            else:
                self.cards_now[card[0]][card[1]] = self.cards_game[card[0]][0]
                del self.cards_game[card[0]][0]
        self.ac_color = "\033[34m"
        self.acted_action = 2
        return True

    # 구매하는 함수
    # ** card 는 [n, m]형태의 리스트로 n = 티어-1 m = 번째 / 4티어 = 킵 카드
    def buy(self, card):
        if card[0] == 3:
            if len(self.my_kept_card[self.turn]) == 0:
                return False
            else:
                if not self.price(self.my_kept_card[self.turn][card[1]]):
                    return False
                else:
                    self.reward += 2 * (self.my_score[self.turn] + 1)
                    del self.my_kept_card[self.turn][card[1]]
        else:
            if self.cards_now[card[0]][card[1]] == 'w0000000':
                return False
            if not self.price(self.cards_now[card[0]][card[1]]):
                return False
            if len(self.cards_game[card[0]]) <= 0:
                self.cards_now[card[0]][card[1]] = 'w0000000'
            else:
                self.cards_now[card[0]][card[1]] = self.cards_game[card[0]][0]
                del self.cards_game[card[0]][0]
        self.ac_color = "\033[32m"
        self.acted_action = 3
        return True

    # 가격 계산 함수
    # buy 에 포함시켜도 되는데 헷갈려서 그냥 함
    def price(self, card_str):
        original_tokens = self.tokens[:]
        ori_token = self.my_token[self.turn][:]
        for i in range(0, 5):
            price = int(card_str[i + 2]) - int(self.my_card[self.turn][i])
            self.used_tokens = price
            if price < 0:
                continue
            if price <= self.my_token[self.turn][i]:
                self.my_token[self.turn][i] -= price
                self.tokens[i] += price
            elif price <= self.my_token[self.turn][i] + self.my_token[self.turn][5]:
                self.my_token[self.turn][5] -= (price - self.my_token[self.turn][i])
                self.tokens[5] += (price - self.my_token[self.turn][i])
                self.tokens[i] += self.my_token[self.turn][i]
                self.my_token[self.turn][i] = 0
            else:
                # print("냥", end="")
                self.tokens = original_tokens
                self.my_token[self.turn] = ori_token
                return False
        self.my_card[self.turn][self.Token[card_str[0]]] += 1
        self.my_score[self.turn] += int(card_str[1])
        return True



    def spit(self, spit):
        t = (lambda x: 1 if x == 0 else 0)(self.turn)
        # t = self.turn
        for i in range(sum(self.my_token[t]) - 10):
            self.my_token[t][spit[i]] -= 1
            self.tokens[spit[i]] += 1
            self.ac_color = "\033[31m"

    def judge(self, q_value):
            # 판단
            for count in range(100):
                if sum(list(reversed(sorted(q_value[:5])))[:3]) < max(q_value[:5]) * 2:
                    oot = 1
                    col_value = max(q_value[:5]) * 2
                else:
                    oot = 3
                    col_value = reduce(lambda x, y: x + y, list(reversed(sorted(q_value[:5])))[0:5])
                if len(list(filter(lambda x: self.tokens[x] > 0, [0, 1, 2, 3, 4]))) == 0:
                    col_value = 3 * (min(q_value))
                for_value = (col_value / 3,
                             max(q_value[5:18]),
                             max(q_value[18:]))
                f_ind = for_value.index(max(for_value))
                # print(f_ind, end="")
                if f_ind == 0:
                    # q_value[:5].index(reversed(sorted(q_value[:5])[0]))
                    if oot == 1:
                        if self.collect(q_value.index(max(q_value[:5]))):
                            self.acted_action = np.argmax(for_value)
                            break
                        else:
                            oot = 3
                    if oot == 3:
                        _q_value = q_value[:5]
                        col_li_ag = [-1, -1, -1]
                        for i in range(3):
                            col_li_ag[i] = _q_value.index(max(_q_value))
                            _q_value[_q_value.index(max(_q_value))] = min(_q_value) - 1
                        if self.collect(col_li_ag[0], col_li_ag[1], col_li_ag[2]):
                            self.acted_action = int(np.argmax(q_value[:5]))
                            break
                        else:
                            for number in range(3):
                                if self.tokens[col_li_ag[number]] <= 0:
                                    q_value[col_li_ag[number]] = min(q_value) - 1
                elif f_ind == 1:
                    if self.buy([int(int(np.argmax(q_value[5:18])) / 4), int(int(np.argmax(q_value[5:18])) % 4)]):
                        self.acted_action = int(np.argmax(q_value[5:18]))
                        break
                    else:
                        q_value[int(np.argmax(q_value[5:18])) + 5] = min(q_value[5:18]) - 1
                elif f_ind == 2:
                    if self.keep(
                            [int(q_value[18:35].index(max(q_value[18:35])) / 5),
                             q_value[18:35].index(max(q_value[18:35])) % 5]):
                        self.acted_action = int(np.argmax(q_value[18:35]))
                        break
                    else:
                        q_value[q_value[18:35].index(max(q_value[18:35])) + 18] = min(q_value) - 1

    # 말그대로 step
    def step(self, q_val=None, action=None, done=False):
        self.reward = 0
        if not q_val is None:
            self.judge(q_val)
        else:
            # TODO q value 를 받은게 아닐때 할 수 있는 액션을 만들어야 하는데 너무 귀찮아 그냥 q value 변환함수를 하나 만드는게 빠를것같아
            pass
        for score in self.my_score:
            if score >= 15 and self.turn == 1:
                self.reward -= 3 * len(self.my_kept_card[self.turn]) + sum(self.my_token[self.turn]) \
                               * math.pow(1.05, (self.dkssud + 10 * self.dkssud2))

                done = True
                print(str(self.dkssud2 * 10 + self.dkssud))
                self.dkssud = 0
                self.dkssud2 = 0
        for lord in self.lord_now:
            lord_list = list(lord)
            del lord_list[0]
            if self.my_card == lord_list:
                self.my_lord[self.turn].append(lord)
                self.lord_now.remove(lord)
                self.my_score[self.turn] += int(lord[0])
                break
        _turn = self.turn
        if self.turn == 0:
            self.turn = 1
        else:
            self.turn = 0

        self.original_score = self.my_score[:]
        if self.turn == 1:
            if self.dkssud >= 10:
                # print(self.my_score)
                self.dkssud = 0
                self.dkssud2 += 1
            else:
                self.dkssud += 1
                # print(self.my_score, end="\t")
            if self.dkssud2 >= 10:
                self.dkssud2 = 0
                self.dkssud = 0
                done = True

        self.reward += 2 * ((self.my_score[_turn] * 10
                             / (lambda x: 1 if x < 1 else x)(self.used_tokens + 1)
                             / (sum(self.my_card[_turn]) + 1)) * math.pow(self.gamma, (self.dkssud + 10 * self.dkssud2))
                            )
        return self.state_return(self.turn), self.reward, done


if __name__ == "__main__":
    pass
