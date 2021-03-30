import vizdoom as vzd

from random import choice
from time import sleep
import tensorflow as tf
import sys, signal
import time
import os
import cv2
from a2c import A2C, PartialRollout
from constants import constants
from env import MyDoom
import numpy as np

unsup = 'action'
envWrap = True
designHead = 'universe'
noReward = False

def init_op(sess):
    variables_to_save = [v for v in tf.global_variables()]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()

    sess.run(init_op)
    sess.run(init_all_op)

def main():
    env = MyDoom()
    agent = A2C(unsup, envWrap, designHead, noReward)

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    
    last_state = env.reset()
    last_features = agent.network.get_initial_features()  # reset lstm memory
    length = 0
    rewards = 0
    values = 0
    ep_bonus = 0
    life_bonus = 0
    timestep_limit = 524 # 2100/4
    episodes = 0
    total_steps = 0

    f_loss = open('./logs/loss.txt', 'a')
    f_pred_loss = open('./logs/pred_loss.txt', 'a')
    f_reward = open('./logs/reward.txt', 'a')

    with tf.Session() as sess, sess.as_default():
        init_op(sess)

        while(True):
            terminal_end = False
            rollout = PartialRollout(True)
            
            for _ in range(constants['ROLLOUT_MAXLEN']):
                # run policy
                fetched = agent.network.act(last_state, *last_features)
                action, value_, features = fetched[0], fetched[1], fetched[2:]

                # run environment: get action_index from sampled one-hot 'action'
                stepAct = action.argmax()
                
                # action repeat
                state, reward, terminal = env.skip_step(actions[stepAct])
                total_steps += 1
                if terminal: state = last_state

                if noReward:
                    reward = 0.

                bonus = agent.ap_network.pred_bonus(last_state, state, action)
                curr_tuple = [last_state, action, reward, value_, terminal, last_features, bonus, state]
                life_bonus += bonus
                ep_bonus += bonus

                # collect the experience
                rollout.add(*curr_tuple)
                rewards += reward
                length += 1
                values += value_[0]

                last_state = state
                last_features = features

                if terminal or length >= timestep_limit:
                    # prints summary of each life if envWrap==True else each game
                    print("Episode %d finished. Sum of shaped rewards: %.2f. Length: %d. Bonus: %.4f." % (episodes ,rewards, length, life_bonus))
                    f_reward.write(str(total_steps) + "," + str(rewards) + "\n")
                    f_loss.flush()
                    f_pred_loss.flush()
                    f_reward.flush()
                    if (episodes % 100 == 0): env.make_gif("./video/" + str(episodes) + ".gif")
                    life_bonus = 0
                    length = 0
                    rewards = 0
                    terminal_end = True
                    last_features = agent.network.get_initial_features()  # reset lstm memory
                    last_state = env.reset()
                    episodes += 1

                if terminal_end:
                    break

            if not terminal_end:
                rollout.r = agent.network.value(last_state, *last_features)

            loss, pred_loss = agent.process(sess, rollout)
            f_loss.write(str(total_steps) + "," + str(loss) + "\n")
            f_pred_loss.write(str(total_steps) + "," + str(pred_loss) + "\n")

    env.close()
    f_reward.close()
    f_loss.close()
    f_pred_loss.close()


if __name__ == "__main__":
    main()