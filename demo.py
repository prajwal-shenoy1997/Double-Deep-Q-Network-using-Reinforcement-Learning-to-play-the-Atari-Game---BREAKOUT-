import gym
from dqn_brain import deepQueueNetwork
from memory import memory
import numpy as np
import os
from collections import deque
import time


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.envs.make("Breakout-v0")
mem=memory(no_slots=1)
dqn=deepQueueNetwork(n_actions=env.action_space.n,memory=mem)
state_queue = deque()

def playing_environment(episodes):
    global env,mem,dqn
    for i in range(episodes):
        _ =env.reset()
        state,*_ = env.step(1)
        done=False
        state_queue.clear()
        state,_ = mem.preprocess(state)
        for _ in range(4):
            state_queue.append(state)
        state_after = np.array(state_queue)
        while done!=True:
            action=dqn.choose_action_for_state(np.reshape(state_after, (1,16)))
            state_, _ , done, info = env.step(action)
            state_,_ = mem.preprocess(state_)
            env.render()
            time.sleep(0.02)
            state_queue.popleft()
            state_queue.append(state_)
            state_after = np.array(state_queue)
    env.close()

if __name__ == '__main__':
    episodes  = 1
    dqn.load_model()
    while True:
        playing_environment(episodes=episodes)
