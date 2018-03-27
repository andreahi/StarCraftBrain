import random
import threading
import time
import traceback

import redis
from keras.models import *

from Actions import get_screen_acions, get_action_map
from Features import get_screen_unit_type, get_available_actions, get_player_data
from Network import Network
from RandomUtils import weighted_random_index
from SC2ENV import SC2Game
from redis_int.RedisUtil import recv_zipped_pickle, send_s

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

RUN_TIME = 300000
THREADS = 20
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 1.0

N_STEP_RETURN = 10100
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = .5
EPS_STOP = .1
EPS_STEPS = 75

MIN_BATCH = 10000

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

counter_lock = threading.Lock()
episode_counter = 0


# ---------
class Brain:
    train_queue = [[], [], [], [], [], [], []]  # s, a, r, s', s' terminal mask
    great_queue = [[], [], [], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.network = Network()
        self.first_run = True
        self.r = redis.StrictRedis(host='192.168.0.25', port=6379, db=0)

    def to_array(self, s, idx):
        return np.array(list(np.array(s)[:, idx]), dtype=np.float32)

    def save(self):
        self.network.save()

    def restore(self):
        self.network.restore()

    def train_push(self, s, a, r, v, s_, rnn_state):
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


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.
        self.rnn_state = self.get_lstm_init_state()
        self.r = redis.StrictRedis(host='192.168.0.25', port=6379, db=0)

    def reset(self):
        # pass
        if random.random() > 0.8:
            self.epsilon = random.random()/2
        else:
            self.epsilon = 0.0
        self.rnn_state = self.get_lstm_init_state()

    def get_lstm_init_state(self):
        c_init = np.zeros((300), np.float32)
        h_init = np.zeros((300), np.float32)
        return [c_init, h_init]

    def getEpsilon(self):
        return self.epsilon
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s, available_actions):
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        a, x_select_point, x_spawningPool, x_spineCrawler, x_Gather, x_extractor, v, rnn_state, a_policy =\
            brain.predict(available_actions, [[s[0]], [s[1]], [s[2]]],
                                              [[self.rnn_state[0]], [self.rnn_state[1]]])

        a = a[0]
        x_select_point = x_select_point[0]
        x_spawningPool = x_spawningPool[0]
        x_spineCrawler = x_spineCrawler[0]
        x_Gather = x_Gather[0]
        x_extractor = x_extractor[0]
        _rnn_state = self.rnn_state
        self.rnn_state = [rnn_state[0][0], rnn_state[1][0]]

        if random.random() < eps:
            a = weighted_random_index(available_actions)
            if a in get_screen_acions():
                x_select_point = np.random.randint(0, 1764)
                x_spawningPool = np.random.randint(0, 1764)
                x_spineCrawler = np.random.randint(0, 1764)
                x_Gather = np.random.randint(0, 1764)
                x_extractor = np.random.randint(0, 1764)

        return a, x_select_point, x_spawningPool, x_spineCrawler, x_Gather, x_extractor,  v, _rnn_state, a_policy

    def get_sample(self, memory, n):
        s, a, _, _, rnn_sate, a_policy = memory[0]
        _, _, _, s_, _, _ = memory[n - 1]

        return s, a, self.R, s_, rnn_sate, a_policy

    def train(self, s, a, r, v, s_, rnn_sate, a_policy):

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        x_select_cats = np.zeros(1764)  # turn action into one-hot representation
        x_spawningPool = np.zeros(1764)  # turn action into one-hot representation
        x_spineCrawler = np.zeros(1764)  # turn action into one-hot representation
        x_Gather = np.zeros(1764)  # turn action into one-hot representation
        x_extractor = np.zeros(1764)  # turn action into one-hot representation
        if a[0] != -1:
            #a_cats = (np.ones(NUM_ACTIONS) * -1) /(NUM_ACTIONS - 1)
            a_cats[a[0]] = 1
        if a[1] != -1:
            x_select_cats[a[1]] = 1
        if a[2] != -1:
            x_spawningPool[a[2]] = 1
        if a[3] != -1:
            x_spineCrawler[a[3]] = 1
        if a[4] != -1:
            x_Gather[a[4]] = 1
        if a[5] != -1:
            x_extractor[a[5]] = 1

        self.memory.append((np.copy(s), np.copy([a_cats, x_select_cats, x_spawningPool, x_spineCrawler, x_Gather, x_extractor]), np.copy(r), s_, rnn_sate, a_policy))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            game_data = []
            _, _, end_r, _, rnn_sate, a_policy = self.get_sample(self.memory, len(self.memory))

            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_, rnn_sate, a_policy = self.get_sample(self.memory, n)
                if r < 0:
                    print("forgetting this game ", r)
                elif r > 99:
                    raise Exception('Should not happen')
                    brain.push_great_game(s, a, r, v, s_, rnn_sate)
                else:
                    if s[1][6 + np.argmax(a[0])] == 0:
                        print(s[1])
                        print(a)
                        brain.train_push(s, a, r*0.99, s_, rnn_sate)
                    else:
                        #if a[1][0] == 1 or a[0][0] == 1:
                        #    send_zipped_pickle(self.r, [s, a, r *1, v, s_, rnn_sate, a_policy], key="trainingsample")
                        #brain.push_great_game(s, a, r, v, s_, rnn_sate)
                        data = [s, a, r, v, s_, rnn_sate, a_policy]
                        #if a[0][0] == 1:
                        #    data = [s, a, r*0.9, v, s_, rnn_sate, a_policy]
                        #send_zipped_pickle(self.r, data, key="trainingsample")
                        game_data.append(data)
                        #brain.train_push(s, a, r, v, s_, rnn_sate)

                time.sleep(0)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            if len(game_data) > 0:
                for _ in range(10):
                    send_s(self.r, game_data, key="gamesample")
            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            raise Exception('Should not happen')

            s, a, r, s_, rnn_sate = self.get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_, rnn_sate)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = SC2Game()
        self.agent = Agent(eps_start, eps_end, eps_steps)
        self.lock_queue = threading.Lock()

    def runEpisode(self):
        # s = self.env.reset()
        #print("epsilon ", self.agent.getEpsilon())

        global counter_lock
        with counter_lock:
            global episode_counter
            episode_counter += 1

        # summary = tf.Summary()
        # summary.value.add(tag='epsilon', simple_value=float(self.agent.getEpsilon()))
        # brain.summary_writer.add_summary(summary, episode_counter)

        self.env.new_episode()
        done, obs, r = self.env.get_state()
        s = self.get_state(obs)
        rnn_state = self.agent.rnn_state

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            # if self.render: self.env.render()

            _available_actions = get_available_actions(obs)
            a, x_select_point, x_spawningPool, x_spineCrawler, x_Gather, x_extractor, v, _rnn_state, a_policy = self.agent.act(s, _available_actions)

            # x = np.random.randint(0, 84)
            # y = np.random.randint(0, 84)
            if _available_actions[a] == 1:
                _a = a
            else:
                print("trid to do invalid action")
                _a = 0

            if a == 2:
                _ = self.env.make_action(_a, x_select_point)
                x_spawningPool = -1
                x_spineCrawler = -1
                x_Gather = -1
                x_extractor = -1
                a = -1

            elif a == 6:
                _ = self.env.make_action(_a, x_spawningPool)
                x_select_point = -1
                x_spineCrawler = -1
                x_Gather = -1
                x_extractor = -1
                a = -1

            elif a == 9:
                _ = self.env.make_action(_a, x_spineCrawler)
                x_select_point = -1
                x_spawningPool = -1
                x_Gather = -1
                x_extractor = -1
                a = -1

            elif a == 10:
                _ = self.env.make_action(_a, x_Gather)
                x_select_point = -1
                x_spawningPool = -1
                x_spineCrawler = -1
                x_extractor = -1
                a = -1

            elif a == 11:
                _ = self.env.make_action(_a, x_extractor)
                x_select_point = -1
                x_spawningPool = -1
                x_spineCrawler = -1
                x_Gather = -1
                a = -1

            else:
                _ = self.env.make_action(_a, -1)
                x_select_point = -1
                x_spawningPool = -1
                x_spineCrawler = -1
                x_Gather = -1
                x_extractor = -1


            done, obs, r = self.env.get_state()

            if done:  # terminal state
                s_ = None
                #print("done and none")
                # s_ = self.get_state(obs)
            else:
                s_ = self.get_state(obs)
            with self.lock_queue:
                   self.agent.train(s, [a, x_select_point, x_spawningPool, x_spineCrawler, x_Gather, x_extractor], r, v, s_, rnn_state, a_policy)

            rnn_state = _rnn_state
            s = s_
            R += r

            if done or self.stop_signal:
                break


    def get_state(self, obs):
        return [np.array( (get_screen_unit_type(obs) == 89), dtype="float32" ), np.concatenate((get_player_data(obs), get_available_actions(obs))), np.array( (get_screen_unit_type(obs) > 0), dtype="float32" )]

    def run(self):
        while not self.stop_signal:
            try:
                self.agent.reset()
                self.runEpisode()
                if os.path.isfile("models/checkpoint"):
                    print("loading model")
                    brain.restore()

            except Exception as exp:
                tb = traceback.format_exc()
                print("got exception while working")
                print(tb)
                print(exp)

    def stop(self):
        self.stop_signal = True


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
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = 7071
NUM_ACTIONS = len(get_action_map())
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment() for _ in range(THREADS)]



for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()


print("Training finished")
env_test.run()
