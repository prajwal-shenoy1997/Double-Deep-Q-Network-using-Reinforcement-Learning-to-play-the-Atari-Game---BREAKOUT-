from collections import deque
import numpy as np
import random
import cv2 as cv
import time

class memory:
    def __init__(self,no_slots):
        self.no_slots=no_slots
        self.mem_counter=0
        self.memory=deque()

    def preprocess(self,state):
        state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
        state = state[93:200, 10:152]
        contours, _ = cv.findContours(state, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        result = []
        error = 0
        for contour in contours:
            M = cv.moments(contour)
            if len(M) == 0:
                continue
            try:
                cnt_X = int(M["m10"] / M["m00"])
            except:
                cnt_X = 0
                error = 1
            try:
                cnt_Y = int(M["m01"] / M["m00"])
            except:
                cnt_Y = 0
                error += 1
            result.append([cnt_X, cnt_Y])
        if len(result) == 2 and result[1][1] != 97:
            result = result[::-1]
        if len(result) == 1 or error == 2:
            return [result[0],result[0]] , True
        return result, False

    def add_to_memory(self,status):
        if self.mem_counter == self.no_slots:
            self.memory.popleft()
            self.mem_counter-=1
        self.memory.append(status)
        self.mem_counter+=1
        return
    def get_batch(self,batch_size):
        self.batch_size=batch_size
        if self.batch_size > self.mem_counter:
            return(None)
        return(random.sample(self.memory,self.batch_size))
    def status(self):
        print(len(self.memory))

if __name__ == '__main__':
    mem=memory(no_slots=2)
    mem.status()
    mem.add_to_memory([1])
    mem.status()
    mem.add_to_memory([2])
    mem.add_to_memory([3])
    print(len(mem.memory))
    print(mem.get_batch(1))
    img=cv.imread("0.jpg",1)
    print(type(img))
    img=mem.preprocess(img)
    cv.imwrite("1.jpg",img)
