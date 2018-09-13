import random
import threading
import time
import traceback

import redis
import numpy as np
from Actions import get_action_map
from Network import Network
from heapq import nlargest

from brain.TensorlowDataHelper import from_indexable
from redis_int.RedisUtil import recv_s, send_s

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_TIME = 300000
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 1.0

N_STEP_RETURN = 10100
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = .5
EPS_STEPS = 750

MIN_BATCH = 10000
TRAINING_SIZE = 15000
BATCH_COUNT = 1

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
        self.r = redis.StrictRedis(host='in.space', port=6379, db=0)
        self.lastTime = time.clock()
        np.set_printoptions(threshold=12)
        #self.features_mean_images = {}

    def optimize(self):
        # if len(self.train_queue[0]) < MIN_BATCH:
        # with self.read_lock:
        self.import_new_data()

        class SceneGenerator(object):
            def __init__(self, trainer):
                self.trainer = trainer

            def __len__(self):
                return BATCH_COUNT

            def __getitem__(self, item):
                a, r, s, v, total_weights, s_f = self.trainer.get_data()
                return a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], r, s[0], s[1], s[2], v, total_weights, s_f[
                    0], s_f[1], s_f[2]

        import tensorflow as tf
        ds = from_indexable(SceneGenerator(self),
                            output_types=(
                            tf.float16, tf.float16, tf.float16, tf.float16, tf.float16, tf.float16, tf.float16,
                            tf.float16, tf.float16,
                            tf.float16, tf.float16, tf.float16, tf.float16, tf.float16, tf.float16, tf.float16,
                            tf.float16, tf.float16),
                            output_shapes=(
                                (TRAINING_SIZE, 12), (TRAINING_SIZE, 1764), (TRAINING_SIZE, 1764),
                                (TRAINING_SIZE, 1764), (TRAINING_SIZE, 1764), (TRAINING_SIZE, 1764),
                                (TRAINING_SIZE, 1024), (TRAINING_SIZE, 1024), (TRAINING_SIZE, 1764),
                                (TRAINING_SIZE, 1),
                                (TRAINING_SIZE, 84, 84), (TRAINING_SIZE, 18), (TRAINING_SIZE, 84, 84),
                                (TRAINING_SIZE, 1),
                                (TRAINING_SIZE, 1),
                                (TRAINING_SIZE, 84, 84),
                                (TRAINING_SIZE, 18),
                                (TRAINING_SIZE, 84, 84)
                            ))

        #self.get_data()
        #it = ds.make_one_shot_iterator()
        #next_element = it.get_next()
        with self.gpu_lock:

            print("getting first losses")

            with tf.Session() as sess:
                datas = []
                for _ in range(BATCH_COUNT):
                    #data = sess.run(next_element)

                    a, r, s, v, total_weights, s_f = self.get_data()
                    data =  a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], r, s[0], s[1], s[2], v, total_weights, \
                           s_f[0], s_f[1], s_f[2]
                    datas.append(data)
                    if len(datas) > 1:
                        del datas[0]
                    for d in datas:
                        v = self.network.predict_value([1] * 15, [d[15], d[16], d[17]], [])
                        # map_predict_loss = self.network.train_predict_map([1] * 13, [d[13], d[14], d[15]], [])
                        # print("map_predict_loss: ", map_predict_loss)
                        v_loss = self.network.train_value(d, v)
                        a_loss = self.network.train_a(d, v)
                        x_loss = self.network.train_select(d, v)
                        x_loss_spawn = self.network.train_spawn(d, v)
                        x_loss_spine = self.network.train_spine(d, v)
                        x_loss_extractor = self.network.train_extractor(d, v)
                        x_loss_gather = self.network.train_gather(d, v)
                        self.network.train_minimap_attack(d, v)
                        self.network.train_minimap_move(d, v)
                        self.network.train_hatchery(d, v)
                        print("v_losss: " + str(v_loss))
                        #total_loss, v_loss, a_loss, x_loss, x_loss_spawn, x_loss_spine, x_loss_gather, x_loss_extractor = self.network.train(d, v)
                #print("total_loss ", total_loss)
                #print("total_loss mean ", np.mean(total_loss))
                print("v_loss ", np.mean(v_loss))
                print("a_loss_policy ", np.mean(a_loss))
                print("x_loss_policy ", np.mean(x_loss))
                print("x_loss_spawn ", np.mean(x_loss_spawn))
                print("x_loss_spine ", np.mean(x_loss_spine))
                print("x_loss_gather ", np.mean(x_loss_gather))
                print("x_loss_extractor ", np.mean(x_loss_extractor))
                del datas

                # losses = self.network.get_losses(a, r, v, np.zeros(shape=(len(v), 1)), s, [], [])
                # print("first_v_loss", np.mean(losses[0]))
                # print("first_a_loss", np.mean(losses[1]))
                # print("first_x_select_loss", np.mean(losses[2]))
                # print("first_x_spawn_loss", np.mean(losses[3]))
                # print("first_x_spine_loss", np.mean(losses[4]))
        print("training done")
        if ((time.clock() - self.lastTime) > 30):
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

    def import_new_data(self):
        train_queue = [[], [], [], [], [], [], [], [], []]
        print("starting optimize")
        for _ in range(1):

            samples = recv_s(self.r, key="gamesample", count=-1)
            for sample in samples:
                for e in sample:
                    self.train_push(train_queue, *e)

        if len(train_queue[0]) > 0:
            self.cache_training_data(train_queue)

    def get_data(self):
        s = [0, 0,
             0]
        s_f = [0, 0,
               0]
        a = [0, 0,
             0, 0,
             0, 0,
             0, 0,
             0]
        cached_samples = recv_s(self.r, key="samplecache", count=TRAINING_SIZE, poplimit=1000000)
        #_ = recv_s(self.r, key="samplecache", poplimit=500000)
        s[0] = np.array(self.to_array(cached_samples, 0) == 342, dtype="float16")
        s[1] = self.to_array(cached_samples, 1)
        s[2] = self.to_array(cached_samples, 2)
        a[0] = self.to_array(cached_samples, 3)
        a[1] = self.to_array(cached_samples, 4)
        a[2] = self.to_array(cached_samples, 5)
        a[3] = self.to_array(cached_samples, 6)
        a[4] = self.to_array(cached_samples, 7)
        a[5] = self.to_array(cached_samples, 8)
        a[6] = self.to_array(cached_samples, 9)
        a[7] = self.to_array(cached_samples, 10)
        a[8] = self.to_array(cached_samples, 11)
        r = self.to_array(cached_samples, 12)
        v = self.to_array(cached_samples, 13)

        s_f[0] = np.array(self.to_array(cached_samples, 15) == 342, dtype="float16")
        s_f[1] = self.to_array(cached_samples, 16)
        s_f[2] = self.to_array(cached_samples, 17)

        # next_s = self.to_array(cached_samples,11)
        if False:
            next_s = np.zeros(shape=(TRAINING_SIZE, 2, 84, 84), dtype="int16")
            for i in range(len(cached_samples)):
                e = cached_samples[i]
                sample = np.zeros(shape=(2, 84, 84))
                sample[:min(e[14].shape[0], 2), :e[14].shape[1]] = e[14][:min(e[14].shape[0], 2)]
                next_s[i] = sample
            if self.first_run:
                print("first run")
                self.first_run = False
            unit_types = set()
            for state_s in next_s:
                for e in state_s:
                    for u in np.unique(e):
                        # print("u: ", u)
                        continue
                        # if not(u == 88 or u == 104):
                        #    continue
                        if u == 0:
                            continue
                        unit_types.add(u)
            feature_images = next_s
            total_weights = np.zeros(shape=(len(unit_types), len(feature_images)), dtype="float16")
            unit_types = list(unit_types)
            print("creating state weights")
            # tmp_compared = np.zeros(shape=next_s.shape)
            for u_index in range(len(unit_types)):
                feature = unit_types[u_index]
                print(feature)
                if feature == 0:
                    continue
                if feature not in self.features_mean_images or random.random() > 0.9:
                    print("feature not found: ", feature)
                    mean_images = []
                    for states in next_s:
                        for future_state in states:
                            bit_feature_image = future_state == feature
                            mean_images.append(bit_feature_image)
                    if feature not in self.features_mean_images:
                        self.features_mean_images[feature] = np.mean(mean_images, axis=0)
                    else:
                        self.features_mean_images[feature] = 0.9 * self.features_mean_images[feature] + 0.1 * np.mean(
                            mean_images, axis=0)
                total_weights[u_index] = np.max(
                    np.mean(
                        np.mean(np.square((self.features_mean_images[feature] - np.equal(next_s, feature))), axis=2),
                        axis=2), axis=1)
            # print("done creating state weights")
            normalized_weights = self.get_weight(total_weights)
            total_weights = normalized_weights
            print("max extra points: ", nlargest(10, total_weights))
            r = r.squeeze() + 0.1 * total_weights
            r = np.expand_dims(r, axis=1)
        return a, r, s, v, np.zeros(shape=(1), dtype="float16"), s_f

    def get_weight(self, total_weights):
        if len(total_weights) == 0:
            return np.zeros(total_weights.shape[1], dtype="float16")
        normalized_weights = []
        for e in total_weights:
            normalized_weights.append((e - np.min(e)) / (np.max(e) - np.min(e) + 0.000001))
        normalized_weights = np.square(np.array(normalized_weights))
        normalized_weights = np.max(normalized_weights, axis=0)
        normalized_weights = ((normalized_weights - np.min(normalized_weights)) / (
                np.max(normalized_weights) - np.min(normalized_weights)))
        return normalized_weights

    def cache_training_data(self, train_queue):
        s, a, r, s_, s_mask, rnn_state, v, a_policy, next_s = train_queue
        print("prepping training data")
        s = [self.to_array(s, 0), self.to_array(s, 1), self.to_array(s, 2)]
        # next_s= map(lambda state: np.array(list(np.array(state)[:, 0]), dtype=np.float32), next_s)
        next_state0 = []
        next_state1 = []
        next_state2 = []
        next_state = []
        for e in next_s:
            next_state.append(self.to_array(e, 0))
            next_state0.append(self.to_array(e, 0))
            next_state1.append(self.to_array(e, 1))
            next_state2.append(self.to_array(e, 2))

        next_s = next_state
        a = [self.to_array(a, 0), self.to_array(a, 1), self.to_array(a, 2),
             self.to_array(a, 3), self.to_array(a, 4), self.to_array(a, 5), self.to_array(a, 6),
             self.to_array(a, 7), self.to_array(a, 8)]
        r = np.vstack(r)
        v = np.vstack(v)
        for i in range(len(train_queue[0])):
            offset = min(len(next_state0[i]) - 1, 1)
            # if 88.0 in s[0][i]:
            #    print("found 88")
            send_s(self.r,
                   [s[0][i], s[1][i], s[2][i], a[0][i], a[1][i], a[2][i], a[3][i], a[4][i], a[5][i], a[6][i], a[7][i],
                    a[8][i], r[i], v[i],
                    next_s[i], next_state0[i][offset], next_state1[i][offset], next_state2[i][offset]],
                   key="samplecache")

    def to_array(self, s, idx):
        return np.array(list(np.array(s)[:, idx]), dtype=np.float16)

    def save(self):
        self.network.save()

    def restore(self):
        self.network.restore()

    def train_push(self, train_queue, s, a, r, v, s_, rnn_state, a_policy, next_s):
        # with self.lock_queue:
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
        train_queue[8].append(np.copy(next_s))

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

time.sleep(1)
print("Starting optimizers")
for o in opts:
    o.start()
    time.sleep(5)
print("Optimizers started")

time.sleep(RUN_TIME)

for o in opts:
    o.stop()
for o in opts:
    o.join()
