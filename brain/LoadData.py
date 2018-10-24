import time

import numpy as np
import redis

from mongo.MongoUtils import retrieve, insert
from redis_int.RedisUtil import recv_pop_s




def to_array(s, idx):
    return np.array(list(np.array(s)[:, idx]), dtype=np.float16)


def cache_training_data(train_queue):
    print("length ", len(train_queue[0]))
    s, a, r, s_, s_mask, rnn_state, v, a_policy, next_s = train_queue
    print("prepping training data")
    print("prepping s")
    s = [to_array(s, 0), to_array(s, 1), to_array(s, 2)]
    next_state0 = []
    next_state1 = []
    next_state2 = []
    print("prepping next_state")
    for e in next_s:
        # next_state.append(self.to_array(e, 0))
        next_state0.append(to_array(e, 0))
        next_state1.append(to_array(e, 1))
        next_state2.append(to_array(e, 2))

    print("prepping a")
    a = [to_array(a, 0), to_array(a, 1), to_array(a, 2),
         to_array(a, 3), to_array(a, 4), to_array(a, 5), to_array(a, 6),
         to_array(a, 7), to_array(a, 8)]
    print("prepping r")
    r = np.vstack(r)
    print("prepping v")
    v = np.vstack(v)
    print("inserting data")
    for i in range(len(train_queue[0])):
        offset = min(len(next_state0[i]) - 1, 1)
        insert(
            [s[0][i], s[1][i], s[2][i], a[0][i], a[1][i], a[2][i], a[3][i], a[4][i], a[5][i], a[6][i], a[7][i],
             a[8][i], r[i], v[i],
             [], next_state0[i][offset], next_state1[i][offset], next_state2[i][offset]]
        )

def train_push(train_queue, s, a, r, v, s_, rnn_state, a_policy, next_s):
    # with self.lock_queue:
    train_queue[0].append(s)
    train_queue[1].append(a)
    train_queue[2].append(r)

    if s_ is None:
        train_queue[3].append(s)
        train_queue[4].append(0.)
    else:
        train_queue[3].append(s_)
        train_queue[4].append(1.)

    train_queue[5].append(rnn_state)
    train_queue[6].append(v)
    train_queue[7].append(a_policy)
    train_queue[8].append(next_s)


def import_new_data(r):
    while True:

        train_queue = [[], [], [], [], [], [], [], [], []]
        print("Importing new data")
        for _ in range(1):

            samples = recv_pop_s(r, key="gamesample", count=1)
            print("len samples ", len(samples))
            for sample in samples:
                for e in sample:
                    train_push(train_queue, *e)

        if len(train_queue[0]) > 0:
            print("caching data ", len(train_queue))
            cache_training_data(train_queue)
        else:
            return 0


def main():
    r = redis.StrictRedis(host='10.0.0.112', port=6379, db=0)
    import_new_data(r)

if __name__ == "__main__":
    main()


