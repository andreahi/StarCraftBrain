import redis


from RedisUtil import recv_zipped_pickle

r = redis.StrictRedis(host='localhost', port=6379, db=0)


import numpy as np
import matplotlib.pyplot as plt



y = []
average = []
count = 0
while True:
    y.append(recv_zipped_pickle(r, key="score"))

    plt.clf()
    plt.scatter(range(len(y)), y)

    average.append(np.average(y[-50:]))
    plt.scatter(range(len(average)), average)
    print "average :", np.average(y[-50:])

    plt.axis()
    plt.ion()

    plt.pause(0.0005)
    count += 1

    if len(y) > 500:
        del y[0]
    if len(average) > 500:
        del average[0]
