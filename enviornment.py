from random import *


class Env:
    def __init__(self):
        self.cards_game = None
        self.cards = (
            ('k011110',
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
             'r500730',),
        )
        self.lords = ('340004',
                      '344000',
                      '333300',
                      '300044',
                      '300333',
                      '304400',
                      '330033',
                      '303330',
                      '300440',
                      '333003',)
        self.lord_now = None
        self.tokens = None
        self.setting()
        return

    # 게임 세팅
    def setting(self):
        self.tokens = [4, 4, 4, 4, 4, 5]
        for i in range(0, 3):
            shuffle(self.cards[i])
        for i in range(0, 3):
            self.cards_game[i] = list(self.cards[i])
            shuffle(self.cards_game[i])
        self.lord_now = sample(self.lords, 3)
        return

    # TODO 상태 크기를 내보내는 함수
    def state_space(self):
        return

    # TODO 상태를 초기화하는 함수
    def state_reset(self):
        return

    # TODO 시작할때의 상태를 나타내는 함수
    def start_set(self):
        return

    # TODO 토큰을 가져오는 함수
    def collect(self):
        return

    # TODO 킵하는 함수
    def keep(self):
        return

    # TODO 구매하는 함수
    def buy(self):
        return

    # TODO 실행가능한 액션의 개수를 반환하는 함수
    def action_space(self):
        return

    # TODO 15점이 넘고 턴이 종료되었는지 확인하는 함수
    def end(self):
        return

    # TODO 액션을 받고 한 턴을 플레이하는 함수
    def step(self):
        return


class GraphicDisplay:
    def __init__(self):
        # 안에 만들어
        return
