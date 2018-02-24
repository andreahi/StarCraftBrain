import random
import threading
import time
import traceback

import redis
import numpy as np
from Actions import get_action_map
from Network import Network
from redis_int.RedisUtil import recv_s

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_TIME = 300000
OPTIMIZERS = 4
THREAD_DELAY = 0.001

GAMMA = 1.0

N_STEP_RETURN = 10100
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = .5
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
    save_lock = threading.Lock()
    def __init__(self):
        self.network = Network()
        self.first_run = True
        self.r = redis.StrictRedis(host='192.168.0.25', port=6379, db=0)
        self.lastTime = time.clock()
        np.set_printoptions(threshold=12)

    def optimize(self):
        #if len(self.train_queue[0]) < MIN_BATCH:
        #with self.read_lock:
        train_queue = [[], [], [], [], [], [], [], []]
        for _ in range(1):
            samples = recv_s(self.r, key="gamesample", count=100)
            RNN_USED = False
            for sample in samples:
                prev_rnn1 = [np.zeros((300), np.float32)]
                prev_rnn2 = [np.zeros((300), np.float32)]
                for e in sample:
                    if random.random() > 1:
                        continue
                    if RNN_USED:
                        out = self.network.predict([1] * 11, [[e[0][0]], [e[0][1]], [e[0][2]]],
                                         [prev_rnn1, prev_rnn2])
                        e[5][0] = np.copy(prev_rnn1[0])
                        e[5][1] = np.copy(prev_rnn2[0])
                    self.train_push(train_queue, *e)
                    if RNN_USED:
                        prev_rnn1 = [out[10][0][0]]
                        prev_rnn2 = [out[10][1][0]]

        s, a, r, s_, s_mask, rnn_state, v, a_policy = train_queue

        print("prepping training data")
        s = [self.to_array(s, 0), self.to_array(s, 1), self.to_array(s, 2)]
        # s = np.stack(s, axis=1)
        a = [self.to_array(a, 0), self.to_array(a, 1), self.to_array(a, 2),
             self.to_array(a, 3), self.to_array(a, 4)]
        r = np.vstack(r)
        v = np.vstack(v)
        a_policy = np.vstack(a_policy)
        # s_ = np.stack(s_, axis=1)
        s_ = [self.to_array(s_, 0), self.to_array(s_, 1)]
        #for i in range(0, len(a)):
        #    if r[i] > v[i]:
        #        a_policy[i][np.argmax(a[0][i])] = 10
        #    else:
        #        a_policy[i][np.argmax(a[0][i])] = -10
        #s_mask = np.vstack(s_mask)
        rnn_state = [self.to_array(rnn_state, 0), self.to_array(rnn_state, 1)]

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
        class_weights = np.square(max(np.sum(a[0]==1, axis=0) ) / (np.sum(a[0]==1, axis=0) + 0.00001))
        class_weights = np.clip(class_weights, 1, 10)
        # _, _, _, v, _ = self.predict(s_, rnn_state)
        # r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        r = r  # set v to 0 where s_ is terminal state
        print("a sum: ", np.sum(a[0] == 1, axis=0))
        print("advantage sum: ", np.sum((r - v) * a[0], axis=0))
        print("rnn : ", rnn_state)

        if self.first_run:
            print("first run")
            self.first_run = False
        with self.gpu_lock:

            losses = self.network.get_losses(a, r, v, np.zeros(shape=(len(v), 1)), s, rnn_state, class_weights, a_policy)
            print("first losses", losses)

        for _ in range(1):
            with self.gpu_lock:
                    total_loss, v_loss, a_loss, x_loss, x_loss_spawn, x_loss_spine = self.network.train(a, r, v, np.zeros(shape=(len(v), 1)), s, rnn_state, class_weights, a_policy, losses)
                    print("total_loss ", total_loss)
                    print("total_loss mean ", np.mean(total_loss))
                    print("v_loss ", np.mean(v_loss))
                    print("a_loss_policy ", np.mean(a_loss))
                    print("x_loss_policy ", np.mean(x_loss))
                    print("x_loss_spawn ", np.mean(x_loss_spawn))
                    print("x_loss_spine ", np.mean(x_loss_spine))
                    print("first_v_loss", np.mean(losses[0]))
                    print("first_a_loss", np.mean(losses[1]))
                    print("first_x_select_loss", np.mean(losses[2]))
                    print("first_x_spawn_loss", np.mean(losses[3]))
                    print("first_x_spine_loss", np.mean(losses[4]))

        if((time.clock() - self.lastTime) > 60):
            print("saving model")
            while 1:
                try:
                    with self.save_lock:
                        brain.save()
                    break
                except Exception as exp:
                    print(exp)
                    time.sleep(1)
            self.lastTime = time.clock()

            # print "optimized done"

    def to_array(self, s, idx):
        return np.array(list(np.array(s)[:, idx]), dtype=np.float16)

    def save(self):
        self.network.save()

    def restore(self):
        self.network.restore()

    def train_push(self, train_queue, s, a, r, v, s_, rnn_state, a_policy):
        #with self.lock_queue:
            train_queue[0].append(np.copy(s))
            train_queue[1].append(np.copy(a))
            train_queue[2].append(np.copy(r))

            if s_ is None:
                train_queue[3].append(np.copy(s))
                train_queue[4].append(0.)
            else:
                train_queue[3].append(np.copy(s_))
                train_queue[4].append(1.)

            train_queue[5].append(np.copy(rnn_state))
            train_queue[6].append(np.copy(v))
            train_queue[7].append(np.copy(a_policy))

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
            try:
                brain.optimize()
            except Exception as exp:
                tb = traceback.format_exc()
                print("got exception while training")
                print(tb)
                print(exp)
            

    def stop(self):
        self.stop_signal = True


# -- main
NUM_STATE = 7071
NUM_ACTIONS = len(get_action_map())
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

opts = [Optimizer() for _ in range(OPTIMIZERS)]

#time.sleep(10000)

for o in opts:
    o.start()
    time.sleep(10)


time.sleep(RUN_TIME)


for o in opts:
    o.stop()
for o in opts:
    o.join()
