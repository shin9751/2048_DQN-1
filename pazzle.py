import os # geckodriverのコピー
import shutil
import numpy as np
import pprint
import copy
import math
import random
import os

import time
import logging


# デバッグ
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.CRITICAL)  # debugしない


SIZE = 4
DIGITS = 4

class Game:

    def __init__(self, board=None, verbose=True):
        board = [row[:] for row in board or [[0] * SIZE] * SIZE]
        self.board = np.array(board)
        self.verbose = verbose
        self.score = 0
        self.reward = 0
        self.terminal = 0
        self.name = os.path.splitext(os.path.basename(__file__))[0]

    def show(self):
        separator = ' -' + '-' * (DIGITS + len(' | ')) * SIZE
        print(separator)
        for row in self.board:
            print(' | ' + ' | '.join(f'{tile:{DIGITS}}' for tile in row) + ' |')
        print(separator)

    def scoring(self, tile):
        self.score += tile * 2
        if self.verbose:
            print(f'{tile}+{tile}={tile*2}')

    def move_left(self):
        self.terminal = False
        self.reward = 0
        #print(self.board)
        for row in self.board:
            for left in range(SIZE - 1):
                for right in range(left + 1, SIZE):
                    if len(row) > 0:
                        #print(right)
                        if row[right] == 0:
                            continue
                        if row[left] == 0:
                            self.reward = 1
                            row[left] = row[right]
                            row[right] = 0
                            self.terminal = True
                            continue
                        else:
                            self.reward = -1
                        if row[left] == row[right]:
                            self.reward = 1
                            self.scoring(row[right])
                            row[left] += row[right]
                            row[right] = 0
                            self.terminal = True
                        else:
                            self.reward = -1 
                            break
        return self.terminal

    def rotate_left(self):
        self.board = [list(row) for row in zip(*self.board)][::-1]

    def rotate_right(self):
        self.board = [list(row)[::-1] for row in zip(*self.board)]

    def rotate_turn(self):
        self.board = [row[::-1] for row in self.board][::-1]

    def flick_left(self):
        self.terminal = self.move_left()
        return self.terminal

    def flick_right(self):
        self.rotate_turn()
        self.terminal = self.move_left()
        self.rotate_turn()
        return self.terminal

    def flick_up(self):
        self.rotate_left()
        self.terminal = self.move_left()
        self.rotate_right()
        return self.terminal

    def flick_down(self):
        self.rotate_right()
        self.terminal = self.move_left()
        self.rotate_left()
        return self.terminal

    def playable(self):
        return any(flick(Game(self.board, verbose=False))
                   for flick in (Game.flick_left, Game.flick_right,
                                 Game.flick_up, Game.flick_down))

    def put_tile(self):
        zeros = [(y, x)
                 for y in range(SIZE)
                 for x in range(SIZE)
                 if self.board[y][x] == 0]
        y, x = random.choice(zeros)
        self.board[y][x] = random.choice((2, 4))

    def read_board(self, browser):  # 現在の局面を取得
        find_tile = browser.find_element_by_class_name('tile-container')\
                       .find_elements_by_class_name
        def tile(x, y):
            try:
                tile = find_tile(f'tile-position-{x}-{y}')
                try:
                    return int(tile[2].text)
                except:
                    return int(tile[0].text)
            except:
                logging.debug(f'{x}-{y}: None')
                return 0
        board = [[tile(x, y) for x in range(1, 5)] for y in range(1, 5)]
        Game(board).show()
        return board

    #状態観測（状況・報酬・結果）
    def observe(self, browser):
        self.board = self.read_board(browser)
        return self.board, self.reward, self.terminal

    #アクション実行
    def execute_action(self, action, html):
        if action == None:
            return
        html.send_keys(action)
