import redis


from RedisUtil import recv_zipped_pickle

r = redis.StrictRedis(host='192.168.0.25', port=6379, db=0)


import numpy as np
import matplotlib.pyplot as plt

USE_GUI = False

y = []
average = []
count = 0
while True:
    y.append(recv_zipped_pickle(r, key="score"))

    if USE_GUI:
        plt.clf()
        plt.scatter(range(len(y)), y)

    average.append(np.average(y[-100:]))
    if USE_GUI:
        plt.scatter(range(len(average)), average)
    print("average :", np.average(y[-100:]), end=" ")
    print("average(20) :", np.average(y[-20:]), end=" ")
    print("std :", np.std(y[-100:]), end=" ")
    print("min :", np.min(y[-100:]), end=" ")
    print("max :", np.max(y[-100:]))

    if USE_GUI:
        plt.axis()
        plt.ion()
        plt.pause(0.0005)

    count += 1

    if len(y) > 500:
        del y[0]
    if len(average) > 500:
        del average[0]
