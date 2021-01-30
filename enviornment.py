from random import *
from functools import reduce
import numpy as np


# 이곳에 있는 모든 토큰은 white blue green red black gold 순으로
# 0 1 2 3 4 5 의 번호를 가짐을 알립니다

class Env:

    def __init__(self):
        self.Token = {'w': 0,
                      'b': 1,
                      'g': 2,
                      'r': 3,
                      'k': 4}
        self.done = False

        self.turn = 0  # 턴 0번플레이어, 1번플레이어로 나뉨

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
                       'r030000',), ('k132200',
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
                                     'r300060',), ('k333530',
                                                   'k433530',
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

        self.cards_game = [[], [], []]  # 순서 없이 섞인 카드들

        self.cards_now = [ [], [], []]  # 현제 나와있는 1 2 3티어 카드
        self.lord_now = None  # 본 판에 나와있는 귀족들
        self.tokens = [4, 4, 4, 4, 4, 5]

        self.my_score = [0, 0]
        self.my_token = [[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]] #내 카드 #상대카
        self.my_card = [[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]] # 내 카드 # 상대 카드
        self.my_kept_card = []  # 내가 찜한 카드
        self.my_lord = [[], []]

        self.action_size = 30

        for i in range(0, 3):
            self.cards_game[i] = list(self.cards[i])
        for i in range(0, 3):
            shuffle(self.cards_game[i])
        for i in range(0, 3):
            self.cards_game[i] = list(self.cards[i])
            shuffle(self.cards_game[i])
            self.cards_now[i] = [self.cards_game[i][:3]]
            del self.cards_game[i][:3]
        self.lord_now = sample(self.lords, 3)
        return

    # TODO 상태를 초기화하는 함수
    def state_reset(self):
        card_now_simple = sum(self.cards_now,[])
        state = [4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #남은 토큰(다이아몬드, 사파이어, 에메랄드, 루비, 줄마노, 찜), 내토큰, 상대토큰
        state_card = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]# 내카드, 상대카
        state.append(state_card)
        for i in range(0,11):
            card = card_now_simple[i]
            state.append(self.Token(card[0])) # 필드 카드 계산
            for j in range(1,7):
                state.append(card[j])
        state_keep_card = np.zeros((1, 42))#내가 들고있는 카드, 상대방이 들고있는 카드 * -1은 모르는 뒷면킵
        state.append(state_keep_card)
        for i in range(0,2):
            lord = self.lord_now[i]
            for j in range(0,5):
                state.append(lord[j])    #귀족
        state_score = [0,0]
        state.append(state_score) #내 점수, 상대 점수
        return state

    def state_return(self):
        return
    # TODO 시작할때의 상태를 나타내는 함수

    # 토큰을 가져오는 함수
    # collect(가져올 토큰:int, 가져올 토큰/필수아님, 가져올 토큰/필수아님)
    def collect(self, *to_collect):
        if len(to_collect) == 1:
            # check collectable
            if self.tokens[to_collect[0]] >= 4:
                self.tokens[self.turn][to_collect[0]] -= 2
                self.my_token[to_collect[0]] += 2
            else:
                return False  # I hope u 2 make AI which can prejudge this thing..or give it as state?
        elif len(to_collect) == 3:
            for jem in to_collect:
                if jem >= 5:
                    return False
                if not self.tokens[jem] == 0:
                    continue
                else:
                    return False
            for jem in to_collect:
                self.tokens[jem] -= 1
                self.my_token[self.turn][jem] += 1
        return True

    # 킵하는 함수
    # ** card 는 [n, m]형태의 리스트로 n = 티어 m = 번째 / 5번째 = 뒷면
    def keep(self, card):
        if len(self.my_kept_card) >= 3:
            return False
        if self.tokens[5] > 0:
            self.tokens[5] -= 1
            self.my_token[self.turn][5] += 1
        if card[1] == 5:
            self.my_kept_card.append(self.cards_game[card[0] - 1][0])
            del self.cards_game[card[0] - 1][0]
        else:
            self.my_kept_card.append(self.cards_now[card[0] - 1][card[1] - 1])
            del self.cards_now[card[0] - 1][card[1] - 1]
            self.cards_now[card[0]][card[1] - 1] = self.cards_game[card[0] - 1][0]
            del self.cards_game[card[0] - 1][card[1] - 1]
        return True

    # 구매하는 함수
    # ** card 는 [n, m]형태의 리스트로 n = 티어 m = 번째 / 4티어 = 킵 카드
    def buy(self, card):
        if card[0] == 4:
            if len(self.my_kept_card[self.turn]) == 0:
                return False
            else:
                self.price(self.my_kept_card[self.turn][card[1]])
        else:
            self.price(self.cards_now[card[0]][card[1]])

        return True

    # 가격 계산 함수
    # buy 에 포함시켜도 되는데 헷갈려서 그냥 함
    def price(self, card_str):
        ori_token = self.my_token[self.turn]
        for i in range(0, 5):
            price = int(card_str[i + 2]) - int(self.my_card[self.turn][i])
            if price <= self.my_token[self.turn][i]:
                self.my_token[self.turn][i] -= price
            elif price <= self.my_token[self.turn][i] + self.my_token[self.turn][5]:
                self.my_token[self.turn][5] -= price - self.my_token[self.turn][i]
                self.my_token[self.turn][i] = 0
            else:
                self.my_token = ori_token
                return False
            self.my_card[self.turn][self.Token[card_str[0]]] += 1
            self.my_score[self.turn] += card_str[1]
        return True

    # TODO 토큰 퉤
    def spit(self, q_value):
        q_value_simple = q_value[:4]
        while True:
            spit_token = np.argmin(q_value_simple[0])
            if self.my_token[spit_token] - 1 < 0:
                del q_value_simple[spit_token]
            else:
                self.my_token[spit_token] = self.my_token[spit_token] - 1
                break
        return None

    # 판단
    def judge(self, q_value):
        while True:
            if reduce(lambda x, y: x + y, list(reversed(sorted(q_value[:4])))[0:3]) <= max(q_value[:4]) * 2:
                oot = 1
                col_value = max(q_value[:4]) * 2
            else:
                oot = 3
                col_value = reduce(lambda x, y: x + y, list(reversed(sorted(q_value[:4])))[0:3])
            for_value = (col_value,
                         max(q_value[5:17]),
                         max(q_value[18:]))

            if for_value.index(max(for_value)) == 0:
                q_value[:4].index(reversed(sorted(q_value[:4])[0]))
                if oot == 1:
                    if self.collect(q_value.index(max(q_value[:4]))):
                        continue
                    else:
                        oot = 3
                if oot == 3:
                    col_li = list(reversed(sorted(q_value[:4])))
                    if self.collect(q_value[:4].index(col_li[0]), q_value[:4].index(col_li[1]),
                                    q_value[:4].index(col_li[2])):
                        continue
                    else:
                        for i in range(0, 5):
                            q_value[i] = 0
            if for_value.index(max(for_value)) == 1:
                if self.buy(q_value[5:17].index(max(q_value[5:17]))):
                    continue
                else:
                    q_value[q_value[5:17].index(max(q_value[5:17])) + 5] = 0
            if for_value.index(max(for_value)) == 2:
                if self.keep(q_value[18:].index(max(q_value[:18]))):
                    continue
                else:
                    q_value[q_value[:18].index(max(q_value[:18])) + 18] = 0
        return

    # TODO 말그대로 step
    def step(self, q_val):
        self.judge(q_val)
        if reduce(lambda x, y: x + y, self.my_token[self.turn]) > 10:
            self.spit()
        for score in self.my_score:
            if score > 15 and self.turn == 1:
                done = True
        for lord in self.lord_now:
            lord_list = list(lord)
            del lord_list[0]
            if self.my_card == lord_list:
                self.my_lord[self.turn].append(lord)
                self.lord_now.remove(lord)
                self.my_score[self.turn] += int(lord[0])
        return self.next_state, self.reward


if __name__ == "__main__":
    env = Env()
    for ch in env.price(env.cards[0][0]):
        print(ch)
