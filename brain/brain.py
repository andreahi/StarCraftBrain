import random
import threading
import time
import traceback

from keras.models import *

from Actions import get_screen_acions, get_action_map
from Features import get_screen_unit_type, get_available_actions, get_player_data
from Network import Network
from RandomUtils import weighted_random_index
from SC2ENV import SC2Game

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RUN_TIME = 300000
THREADS = 50
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 1.0

N_STEP_RETURN = 10100
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = .5
EPS_STOP = .2
EPS_STEPS = 750

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

    def optimize(self):
        #if len(self.train_queue[0]) < MIN_BATCH:
        if len(self.train_queue[0]) < 1000 + 100:
            time.sleep(1)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < 1000 + 100:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask, rnn_state, v = self.train_queue
            self.train_queue = [[], [], [], [], [], [], []]
            self.train_queue = copy.deepcopy(self.great_queue)

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
        s = [self.to_array(s, 0), self.to_array(s, 1)]
        # s = np.stack(s, axis=1)
        a = [self.to_array(a, 0), self.to_array(a, 1), self.to_array(a, 2), self.to_array(a, 3), self.to_array(a, 4), self.to_array(a, 5), self.to_array(a, 6), self.to_array(a, 7), self.to_array(a, 8)]
        r = np.vstack(r)
        v = np.vstack(v)
        # s_ = np.stack(s_, axis=1)
        s_ = [self.to_array(s_, 0), self.to_array(s_, 1)]
        s_mask = np.vstack(s_mask)
        rnn_state = [self.to_array(rnn_state, 0), self.to_array(rnn_state, 1)]

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
        class_weights = np.square(max(np.sum(a[0]==1, axis=0) ) / (np.sum(a[0]==1, axis=0) + 0.00001))
        class_weights = np.clip(class_weights, 0, 5000)
        # _, _, _, v, _ = self.predict(s_, rnn_state)
        # r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        r = r  # set v to 0 where s_ is terminal state
        print("#great games ", len(self.great_queue[0]))
        print("a sum: ", np.sum(a[0] == 1, axis=0))
        print("advantage sum: ", np.sum((r - v) * a[0], axis=0))
        if self.first_run:
            print("first run")
            for _ in range(100):
                v_loss = self.network.train_value(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)
                print("loss_value ", np.mean(v_loss))
                self.first_run = False

        for _ in range(10):
            #time.sleep(0.1)
            v_loss = 0.0
            for _ in range(0):
                a_loss, x_loss, y_loss, v_loss = self.network.train(a, r, r, np.ones(shape=(len(v), 1)), s, rnn_state, class_weights)
            print("loss_value2 ", np.mean(v_loss))

            a_loss, x_loss, y_loss, v_loss = self.network.train(a, r, v, np.zeros(shape=(len(v), 1)), s, rnn_state, class_weights)
            print("a_loss_policy ", np.mean(a_loss))
            print("x_loss_policy ", np.mean(x_loss))
            print("y_loss_policy ", np.mean(y_loss))
            print("loss_value ", np.mean(v_loss))


            # print "optimized done"

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

    def reset(self):
        # pass
        self.rnn_state = self.get_lstm_init_state()

    def get_lstm_init_state(self):
        c_init = np.zeros((100), np.float32)
        h_init = np.zeros((100), np.float32)
        return [c_init, h_init]

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s, available_actions):
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        a, x_select_point, y_select_point, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather, v, rnn_state =\
            brain.predict(available_actions, [[s[0]], [s[1]]],
                                              [[self.rnn_state[0]], [self.rnn_state[1]]])
        _rnn_state = self.rnn_state
        self.rnn_state = [rnn_state[0][0], rnn_state[1][0]]

        if random.random() < eps:
            a = weighted_random_index(available_actions)
            if a in get_screen_acions():
                x_select_point = np.random.randint(0, 84)
                y_select_point = np.random.randint(0, 84)
                x_spawningPool = np.random.randint(0, 84)
                y_spawningPool = np.random.randint(0, 84)
                x_spineCrawler = np.random.randint(0, 84)
                y_spineCrawler = np.random.randint(0, 84)
                x_Gather = np.random.randint(0, 84)
                y_Gather = np.random.randint(0, 84)
            return a, x_select_point, y_select_point, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather,  v, _rnn_state

        else:
            return a, x_select_point, y_select_point, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather,  v, _rnn_state

    def get_sample(self, memory, n):
        s, a, _, _, rnn_sate = memory[0]
        _, _, _, s_, _ = memory[n - 1]

        return s, a, self.R, s_, rnn_sate

    def train(self, s, a, r, v, s_, rnn_sate):

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        x_select_cats = np.zeros(84)  # turn action into one-hot representation
        y_select_cats = np.zeros(84)  # turn action into one-hot representation
        x_spawningPool = np.zeros(84)  # turn action into one-hot representation
        y_spawningPool = np.zeros(84)  # turn action into one-hot representation
        x_spineCrawler = np.zeros(84)  # turn action into one-hot representation
        y_spineCrawler = np.zeros(84)  # turn action into one-hot representation
        x_Gather = np.zeros(84)  # turn action into one-hot representation
        y_Gather = np.zeros(84)  # turn action into one-hot representation
        if a[0] != -1:
            #a_cats = (np.ones(NUM_ACTIONS) * -1) /(NUM_ACTIONS - 1)
            a_cats[a[0]] = 1
        if a[1] != -1:
            x_select_cats[a[1]] = 1
        if a[2] != -1:
            y_select_cats[a[2]] = 1
        if a[3] != -1:
            x_spawningPool[a[3]] = 1
        if a[4] != -1:
            y_spawningPool[a[4]] = 1
        if a[5] != -1:
            x_spineCrawler[a[5]] = 1
        if a[6] != -1:
            y_spineCrawler[a[6]] = 1
        if a[7] != -1:
            x_Gather[a[7]] = 1
        if a[8] != -1:
            y_Gather[a[8]] = 1

        self.memory.append((np.copy(s), np.copy([a_cats, x_select_cats, y_select_cats, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather]), np.copy(r), s_, rnn_sate))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            _, _, end_r, _, rnn_sate = self.get_sample(self.memory, len(self.memory))
            if r > 20:
                print("I did geat! ", r)
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_, rnn_sate = self.get_sample(self.memory, n)
                if r > 99:
                    raise Exception('Should not happen')
                    brain.push_great_game(s, a, r, v, s_, rnn_sate)
                else:
                    if s[1][5 + np.argmax(a[0])] == 0:
                        print(s[1])
                        print(a)
                        brain.train_push(s, a, r*0.99, s_, rnn_sate)
                    else:
                        #brain.push_great_game(s, a, r, v, s_, rnn_sate)
                        brain.train_push(s, a, r, v, s_, rnn_sate)

                time.sleep(0.1)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

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

    def runEpisode(self):
        # s = self.env.reset()
        print("epsilon ", self.agent.getEpsilon())

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
            a, x_select_point, y_select_point, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather, v, _rnn_state = self.agent.act(s, _available_actions)

            # x = np.random.randint(0, 84)
            # y = np.random.randint(0, 84)
            if _available_actions[a] == 1:
                _a = a
            else:
                _a = 0

            if a == 2:
                _ = self.env.make_action(_a, x_select_point, y_select_point)
                x_spawningPool = -1
                y_spawningPool = -1
                x_spineCrawler = -1
                y_spineCrawler = -1
                x_Gather = -1
                y_Gather = -1

            elif a == 6:
                _ = self.env.make_action(_a, x_spawningPool, y_spawningPool)
                x_select_point = -1
                y_select_point = -1
                x_spineCrawler = -1
                y_spineCrawler = -1
                x_Gather = -1
                y_Gather = -1

            elif a == 9:
                _ = self.env.make_action(_a, x_spineCrawler, y_spineCrawler)
                x_select_point = -1
                y_select_point = -1
                x_spawningPool = -1
                y_spawningPool = -1
                x_Gather = -1
                y_Gather = -1

            elif a == 10:
                _ = self.env.make_action(_a, x_Gather, y_Gather)
                x_select_point = -1
                y_select_point = -1
                x_spawningPool = -1
                y_spawningPool = -1
                x_spineCrawler = -1
                y_spineCrawler = -1

            else:
                _ = self.env.make_action(_a, -1, -1)

            done, obs, r = self.env.get_state()

            if done:  # terminal state
                s_ = None
                # s_ = self.get_state(obs)
            else:
                s_ = self.get_state(obs)

            self.agent.train(s, [a, x_select_point, y_select_point, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather], r, v, s_, rnn_state)

            rnn_state = _rnn_state
            s = s_
            R += r

            if done or self.stop_signal:
                break


    def get_state(self, obs):
        return [get_screen_unit_type(obs) / 500.0, np.concatenate((get_player_data(obs), get_available_actions(obs)))]

    def run(self):
        while not self.stop_signal:
            try:
                self.agent.reset()
                self.runEpisode()
                brain.save()

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
opts = [Optimizer() for _ in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
env_test.run()
