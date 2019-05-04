import gym
from dqn_brain import deepQueueNetwork
from memory import memory
import numpy as np
import os
from collections import deque
import time
"""
new one 
memory.py
tryinh.py
mess.py
"""

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.envs.make("Breakout-v0")
mem=memory(no_slots=10000)
dqn=deepQueueNetwork(n_actions=env.action_space.n,memory=mem)
state_queue = deque()

def playing_environment(episodes,steps):
    global env,mem,dqn
    for i in range(episodes):
        state=env.reset()
        state,*_ = env.step(1)
        done=False
        state_queue.clear()
        state,_ = mem.preprocess(state)
        for _ in range(4):
            state_queue.append(state)
        state_after = np.array(state_queue)
        desc = True
        dead = False
        while done!=True:
            action=dqn.choose_action(np.reshape(state_after, (1,16)))
            state_, _ , done, info = env.step(action)
            state_,redundant = mem.preprocess(state_)
            #env.render()
            state_before = np.array(state_queue)
            state_queue.popleft()
            state_queue.append(state_)
            state_after = np.array(state_queue)
            reward = 0.0
            if redundant == False:
                dead =False
            if state[0][1] > 98 or (state[0][1]>=97 and (abs(state[0][0]-state[1][0])>15)):
                reward = -500.0
                dead = True
            elif (state[0][1] - state_[0][1]) > 0 and desc == True and state_[0][1]>80:
                reward = 1000.0
            elif redundant==True and desc==True and dead==True:
                reward = -10.0
            if (state_[0][1] - state[0][1]) > 0 and desc == False:
                desc = True
            if desc == True:
                if reward== 1000.0:
                    desc=False
                mem.add_to_memory(np.array((np.reshape(state_before, (16,)), action, reward, np.reshape(state_after, (16,)), done)))
            dqn.learn(global_step=11000 + i)
            state = state_.copy()
        print("End of episode {}".format(i+1))

    env.close()

if __name__ == '__main__':
    episodes  = 1000
    tracker = 1
    dqn.load_model()
    while True:
        dqn.epsilon = 0.25
        dqn.counter = 0
        playing_environment(episodes=episodes,steps=tracker)
        if input("Wanna continue?")=='no':
            break
        else:
            tracker+=1
    dqn.save()
