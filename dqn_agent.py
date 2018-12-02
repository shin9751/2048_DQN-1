from selenium.webdriver.common.keys import Keys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
#from keras.initializers import zeros
import numpy as np
import random
import os
import tensorflow as tf
from collections import deque
from keras import backend as K


from pazzle import Game

# [1]損失関数の定義
# 損失関数にhuber関数を使用 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

#Keyフリック定義
KEY_FLICK = {
    Keys.RIGHT: Game.flick_right,
    Keys.DOWN: Game.flick_down,
    Keys.LEFT: Game.flick_left,
    Keys.UP: Game.flick_up,
}

#DQNエージェントクラス
class agent:
    #イニシャライズでKeras起動
    def __init__(self, environment_name, learning_rate=0.01, state_size=16, action_size=4, hidden_size=10):
        # モデルを生成する
        #入力:マス目縦横4マスの4次元
        #出力：上下左右へのアクション4次元
        self.model = Sequential()
        # 全結合層(4層->10層)
        self.model.add(Dense(input_dim=4, output_dim=16))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        #0で初期化
        #self.model.add(Dense(4, kernel_initializer=zeros()))
        # 全結合層(10層->4層)
        self.model.add(Dense(output_dim=4))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

        max_size=1000
        self.buffer = deque(maxlen=max_size)
        self.exploration = 0.1

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = KEY_FLICK

    # 学習のためのデータ。
    # 今回、Xは[0,0]または[1,1]の2種類。
    # Yは0または1の2種類
    # X:[0,0] => Y:0
    # X:[1,1] => Y:1
    # という対応になっている
    """
    X_list = [[0, 0], [1, 1], [0, 0], [1, 1], [1, 1], [1, 1]]
    Y_list = [   [0],    [1],    [0],    [1],    [1],    [1]]

    # kerasのmodelに渡す前にX,Yをnumpyのarrayに変換する。
    X = np.array(X_list)
    Y = np.array(Y_list)
    """
    #tmpに格納
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size)
        return [self.buffer[ii] for ii in idx]

    # 重みの学習
    def replay(self, batch_size, gamma):
        #input/targeetは同じでなければならない
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 4))
        #targets = np.array([[[0 for i in range(4)] for j in range(4)], 4])
        mini_batch = self.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            #print(state_b[i:i + 1]) #表示成功
            #print(action_b)
            #print(reward_b)
            #print(next_state_b) #表示成功
            inputs[i:i + 1] = state_b[i:i + 1]
            #target = reward_b
            #print(inputs) #表示成功
            """
            if not np.allclose(state_b, next_state_b):
                next_state_b = np.reshape(next_state_b, [4, 4])
                #print(next_state_b) #成功
                retmainQs = self.model.predict(next_state_b)
                #print(retmainQs) #成功
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                #print(next_action) #成功
                target = reward_b + gamma * self.model.predict(next_state_b)[next_action]
                #print(target) #4
            """
            """
            state_b = np.reshape(state_b, [4, 4])
            targets = self.model.predict(state_b)
            """
            if reward_b >= 0:
                targets[i:i + 1] = next_state_b[i:i + 1]
            else:
                targets[i:i + 1] = 0
            #targets = targets * target                 #教師信号
            #inputs_train = np.append(inputs, 0)
            #print(targets)
            self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

    #tmpに追加
    def add(self, experience):
        self.buffer.append(experience)

    #バッファの長さ
    def len(self):
        return len(self.buffer)

    def calculate_gain(self, board):  # ある局面が与えられた時に、その評価値を計算する
        """
        評価関数の中身
        ①右下に大きい数字が来るようにしたい
        ②上方向は動かさない（他に手がない場合を除く）
        ③空白は多い方がいい
        ④ジグザグに重みをつける
        """
        spots = ((4, 4, 800 * 8), (3, 4, 750 * 8), (2, 4, 700 * 8), (1, 4, 650 * 8),
                 (1, 3, 600 * 2), (2, 3, 550 * 2), (3, 3, 500 * 2), (4, 3, 450 * 2),
                 (1, 2, 1), (2, 2, 1), (3, 2, 1), (4, 2, 1))
        gain = sum(board[y-1][x-1] * point for x, y, point in spots)
        gain += sum(100 * (tile == 0) for row in board for tile in row)
        gain += sum(1000 * (a < b) for a, b in zip(*board[-2:]))     #下1,2段
        gain += sum(5000 * (a == b) for a, b in zip(*board[-2:]))    #下1,2段
        gain += sum(5000 * (a == b) for a, b in zip(*board[-3:-1]))  #下2,3段
        return gain

    def get_each_gain(self, board):  # それぞれの方向の評価値を計算する
        def gain(flick):
            game = Game(board)
            return flick(game) and self.calculate_gain(game.board) or 0
        return {key: gain(flick) for key, flick in self.enable_actions.items()}

    def randint(self, n):
        return random.randint(0, n - 1)

    #学習結果から最適な手を計算・取得する
    def select_action(self, state):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        exploration = 0.1

        if np.random.rand() >= exploration:
            #print(state)
            state = np.reshape(state, [4, 4])
            state_t = self.model.predict(state)  # 各方向の評価関数を計算する
            #print(state_t)
            state_g = [[0 for i in range(4)] for j in range(4)]
            for i in range(4):
                for j in range(4):
                    state_g[i][j] = int(state_t[i][j])
            #print(state_g)
            gain = self.get_each_gain(state_g)
            #print(gain)
            print('評価値:', list(gain.values()))
            #print('学習')
            if sum(gain.values()) == 0:
                return None  # Game Over
            # どの方向も同じ価値を持つ場合硬直してしまうので、乱数を足してほぐす
            if gain[Keys.RIGHT] == gain[Keys.DOWN] == gain[Keys.LEFT] == gain[Keys.UP]:
                gain[Keys.RIGHT] += self.randint(100)
                gain[Keys.LEFT] += self.randint(100)
            if gain[Keys.RIGHT] == gain[Keys.DOWN]:
                gain[Keys.RIGHT] += self.randint(10)
                gain[Keys.DOWN] += self.randint(10)
            if gain[Keys.RIGHT] == gain[Keys.LEFT]:
                gain[Keys.RIGHT] += self.randint(10)
                gain[Keys.LEFT] += self.randint(10)
            if gain[Keys.DOWN] == gain[Keys.LEFT]:
                gain[Keys.DOWN] += self.randint(10)
                gain[Keys.LEFT] += self.randint(10)
            return max(gain.keys(), key=lambda key: gain[key])

        else:
            #print('ランダム')
            #print(state)
            return random.choice((Keys.RIGHT, Keys.DOWN, Keys.LEFT))  # ランダムに行動する
