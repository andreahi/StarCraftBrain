import random
import threading
import time
import traceback

import redis
from keras.models import *

from brain.Actions import get_screen_acions, get_action_map
from brain.Features import get_screen_unit_type, get_available_actions, get_player_data
from brain.Network import Network
from brain.RandomUtils import weighted_random_index
from brain.SC2ENV import SC2Game
from redis_int.RedisUtil import recv_zipped_pickle, send_zipped_pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_TIME = 300000
OPTIMIZERS = 2
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
    train_queue = [[], [], [], [], [], [], [], []]  # s, a, r, s', s' terminal mask
    great_queue = [[], [], [], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    read_lock = threading.Lock()
    gpu_lock = threading.Lock()

    def __init__(self):
        self.network = Network()
        self.first_run = True
        self.r = redis.StrictRedis(host='192.168.0.25', port=6379, db=0)

    def optimize(self):
        #if len(self.train_queue[0]) < MIN_BATCH:
        with self.read_lock:
            while len(self.train_queue[0]) < 500:
                sample = recv_zipped_pickle(self.r, key="trainingsample")
                self.train_push(*sample)


            s, a, r, s_, s_mask, rnn_state, v, a_policy = self.train_queue
            self.train_queue = [[], [], [], [], [], [], [], []]
            #self.train_queue = copy.deepcopy(self.great_queue)

            GREAT_GAME_SIZE = 1000
            if len(self.great_queue[0])> GREAT_GAME_SIZE:
                del self.great_queue[0][0:len(self.great_queue[0]) - GREAT_GAME_SIZE]
                del self.great_queue[1][0:len(self.great_queue[1]) - GREAT_GAME_SIZE]
                del self.great_queue[2][0:len(self.great_queue[2]) - GREAT_GAME_SIZE]
                del self.great_queue[3][0:len(self.great_queue[3]) - GREAT_GAME_SIZE]
                del self.great_queue[4][0:len(self.great_queue[4]) - GREAT_GAME_SIZE]
                del self.great_queue[5][0:len(self.great_queue[5]) - GREAT_GAME_SIZE]
                del self.great_queue[6][0:len(self.great_queue[6]) - GREAT_GAME_SIZE]

        print("prepping training data")
        s = [self.to_array(s, 0), self.to_array(s, 1), self.to_array(s, 2)]
        # s = np.stack(s, axis=1)
        a = [self.to_array(a, 0), self.to_array(a, 1), self.to_array(a, 2), self.to_array(a, 3), self.to_array(a, 4), self.to_array(a, 5), self.to_array(a, 6), self.to_array(a, 7), self.to_array(a, 8)]
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
        print("#great games ", len(self.great_queue[0]))
        print("a sum: ", np.sum(a[0] == 1, axis=0))
        print("advantage sum: ", np.sum((r - v) * a[0], axis=0))
        print("rnn : ", rnn_state)

        with self.gpu_lock:

            if self.first_run:
                print("first run")
                self.first_run = False
                for _ in range(10):
                    v_loss = self.network.train_value(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)
                    print("loss_value ", np.mean(v_loss))


            for _ in range(0):
                v_loss2 = self.network.train_value(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)
                print("loss_value2 ", np.mean(v_loss2))

            for _ in range(1):
                #time.sleep(0.1)
                #v_loss = 0.0
                for _ in range(0):
                    a_loss, x_loss, y_loss, v_loss2 = self.network.train(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)

                a_loss, x_loss, y_loss = self.network.train(a, r, v, np.zeros(shape=(len(v), 1)), s, rnn_state, class_weights, a_policy)
                print("a_loss_policy ", np.mean(a_loss))
                print("x_loss_policy ", np.mean(x_loss))
                print("y_loss_policy ", np.mean(y_loss))

            for _ in range(1):
                v_loss2 = self.network.train_value(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)
                print("loss_value2 ", np.mean(v_loss2))

            brain.save()

            # print "optimized done"

    def to_array(self, s, idx):
        return np.array(list(np.array(s)[:, idx]), dtype=np.float32)

    def save(self):
        self.network.save()

    def restore(self):
        self.network.restore()

    def train_push(self, s, a, r, v, s_, rnn_state, a_policy):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(s)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

            self.train_queue[5].append(rnn_state)
            self.train_queue[6].append(v)
            self.train_queue[7].append(a_policy)

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
