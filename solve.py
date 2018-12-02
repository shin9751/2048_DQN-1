from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import time
import logging
import random
import numpy as np

from pazzle import Game
from dqn_agent import agent

def play(browser, html):
    elem_before_text = -1
    scores = []  # スコア推移
    last_score = -1
    WAIT = 0.3

    # parameters
    n_epochs = 1000
    batch_size = 4
    gamma = 0.99

    # environment, agent
    env = Game()
    dqn = agent(env.name)

    # variables
    win = 0

    for i in range(n_epochs):
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        state_t_1, reward_t, terminal = env.observe(browser)
        print(f'{i}手目：')

        while not terminal:
            time.sleep(WAIT)
            state_t = state_t_1
            #print(state_t)

            # execute action in environment
            action_t = dqn.select_action(state_t)
            env.execute_action(action_t, html)
            action_flag = True
            if action_t is None:
                action_flag = False

            # observe environment
            state_t_1, reward_t, terminal = env.observe(browser)
            dqn.add((state_t, action_t, reward_t, state_t_1))     # メモリの更新する

            # experience replay
            dqn.replay(batch_size, gamma)
            if not action_flag:
                break

            score = browser.find_element_by_class_name('score-container').text
            if score == last_score:
                html.send_keys(random.choice((Keys.RIGHT, Keys.DOWN, Keys.LEFT)))
                time.sleep(WAIT)
            print('スコア:', score)
            scores.append(score)
            last_score = score

    logging.debug('End of program')

if __name__ == '__main__':
    # 2048のページを開く
    #browser = webdriver.Chrome() #chromeは何故か動作しない
    browser = webdriver.Firefox() #本来はインスコパス入れるけど入れたら動かなかった
    browser.get('https://gabrielecirulli.github.io/2048/')

    # ゲーム開始の準備
    score = []# スコアの推移
    html = browser.find_element_by_tag_name('html')

    logging.debug('Start of program')
    play(browser, html)
    logging.debug('End of program')
