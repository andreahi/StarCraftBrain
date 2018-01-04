import threading
import time

import redis
import numpy as np
from Actions import get_action_map
from Network import Network
from redis_int.RedisUtil import recv_s

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_TIME = 300000
OPTIMIZERS = 3
THREAD_DELAY = 0.001

GAMMA = 1.0

N_STEP_RETURN = 10100
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = .5
EPS_STOP = .01
EPS_STEPS = 750

MIN_BATCH = 10000

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

counter_lock = threading.Lock()
episode_counter = 0


# ---------
class Brain:
    lock_queue = threading.Lock()
    read_lock = threading.Lock()
    gpu_lock = threading.Lock()

    def __init__(self):
        self.network = Network()
        self.first_run = True
        self.r = redis.StrictRedis(host='192.168.0.25', port=6379, db=0)
        self.lastTime = time.clock()


    def optimize(self):
        #if len(self.train_queue[0]) < MIN_BATCH:
        #with self.read_lock:
        train_queue = [[], [], [], [], [], [], [], []]
        samples = recv_s(self.r, key="gamesample", count=20)
        p1 = self.network.predict([1] * 11, [[samples[0][0][0][0]], [samples[0][0][0][1]], [samples[0][0][0][2]]],
                                       [[samples[0][0][5][0]], [samples[0][0][5][1]]])

        for sample in samples:
            self.train_push(train_queue, *sample)

        s, a, r, s_, s_mask, rnn_state, v, a_policy = train_queue

        print("prepping training data")
        s = np.array([self.to_array(s, 0), self.to_array(s, 1), self.to_array(s, 2)])
        # s = np.stack(s, axis=1)
        a = np.array([self.to_array(a, 0), self.to_array(a, 1), self.to_array(a, 2), self.to_array(a, 3), self.to_array(a, 4), self.to_array(a, 5), self.to_array(a, 6), self.to_array(a, 7), self.to_array(a, 8)])
        r = np.vstack(r)
        v = np.vstack(v)
        a_policy = np.vstack(a_policy)
        # s_ = np.stack(s_, axis=1)
        s_ = [self.to_array(s_, 0), self.to_array(s_, 1)]
        for i in range(0, len(a)):
            if r[i] > v[i]:
                a_policy[i][np.argmax(a[0][i])] = 10
            else:
                a_policy[i][np.argmax(a[0][i])] = -10
        s_mask = np.vstack(s_mask)
        rnn_state = [self.to_array(rnn_state, 0), self.to_array(rnn_state, 1)]

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
        class_weights = np.square(max(np.sum(a[0]==1, axis=0) ) / (np.sum(a[0]==1, axis=0) + 0.00001))
        class_weights = np.clip(class_weights, 1, 100)
        # _, _, _, v, _ = self.predict(s_, rnn_state)
        # r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        r = r  # set v to 0 where s_ is terminal state
        print("a sum: ", np.sum(a[0] == 1, axis=0))
        print("advantage sum: ", np.sum((r - v) * a[0], axis=0))
        print("rnn : ", rnn_state)

        with self.gpu_lock:

            if self.first_run:
                print("first run")
                self.first_run = False
                for _ in range(10):
                    idx = np.random.randint(len(a), size=1000)
                    v_loss = self.network.train_value(a[idx], r[idx], r[idx], np.ones(shape=(len(v), 1)), s[idx], rnn_state[idx], class_weights)
                    print("loss_value ", np.mean(v_loss))


            for _ in range(100):
                idx = np.random.randint(len(a), size=1000)

                v_loss2 = self.network.train_value(a[idx], r[idx], r[idx], np.ones(shape=(len(v), 1)), s[idx], rnn_state[idx], class_weights)
                print("loss_value2 ", np.mean(v_loss2))

            for _ in range(1):
                #time.sleep(0.1)
                #v_loss = 0.0
                for _ in range(0):
                    a_loss, x_loss, y_loss, v_loss2 = self.network.train(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)
                idx = np.random.randint(len(a), size=1000)
                a_loss, x_loss, y_loss = self.network.train(a[idx], r[idx], v[idx], np.zeros(shape=(len(v), 1)), s[idx], rnn_state, class_weights, a_policy)
                print("a_loss_policy ", np.mean(a_loss))
                print("x_loss_policy ", np.mean(x_loss))
                print("y_loss_policy ", np.mean(y_loss))

            for _ in range(0):
                v_loss2 = self.network.train_value(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)
                print("loss_value2 ", np.mean(v_loss2))
            if((time.clock() - self.lastTime) > 60):
                print("saving model")
                brain.save()
                self.lastTime = time.clock()

            # print "optimized done"

    def to_array(self, s, idx):
        return np.array(list(np.array(s)[:, idx]), dtype=np.float32)

    def save(self):
        self.network.save()

    def restore(self):
        self.network.restore()

    def train_push(self, train_queue, s, a, r, v, s_, rnn_state, a_policy):
        with self.lock_queue:
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

    def push_great_game(self, s, a, r, v, s_, rnn_state):
        with self.lock_queue:
            self.great_queue[0].append(s)
            self.great_queue[1].append(a)
            self.great_queue[2].append(r)

            if s_ is None:
                self.great_queue[3].append(s)
                self.great_queue[4].append(0.)
            else:
                self.great_queue[3].append(s_)
                self.great_queue[4].append(1.)
            self.great_queue[5].append(rnn_state)
            self.great_queue[6].append(v)


    def predict(self, available_actions, s, batch_rnn_state):
        return self.network.predict(available_actions, s, batch_rnn_state)


# ---------
frames = 0




# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()
            
            

    def stop(self):
        self.stop_signal = True


# -- main
NUM_STATE = 7071
NUM_ACTIONS = len(get_action_map())
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

opts = [Optimizer() for _ in range(OPTIMIZERS)]

for o in opts:
    o.start()


time.sleep(RUN_TIME)


for o in opts:
    o.stop()
for o in opts:
    o.join()
