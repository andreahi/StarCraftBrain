import random

import tensorflow as tf
from sklearn.preprocessing import normalize

from tensorflow.contrib import slim

import numpy as np
from tensorflow.python.keras._impl.keras.layers import LeakyReLU
from tensorflow.python.training.optimizer import _OptimizableVariable


class Network:
    NUM_ACTIONS = 11
    NUM_COORDS_X = 84
    NUM_COORDS_Y = 84

    INPUT_IMAGE = 84
    NUM_SINGLE = 17

    LEARNING_RATE = 1e-7
    LOSS_V = .1  # v loss coefficient
    LOSS_ENTROPY = .001 # entropy coefficient
    WEIGHT_DECAY = 0.000
    def __init__(self):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        np.set_printoptions(precision=3)

        self._build_model()
        self._build_graph()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.session.run(tf.global_variables_initializer())
        self.restore()

        self.default_graph = tf.get_default_graph()

        #self.summary_writer = tf.summary.FileWriter("train", graph=self.default_graph)

        self.default_graph.finalize()  # avoid modifications


    #@tf.RegisterGradient("CustomGrad5")
    #def _const_mul_grad(unused_op, grad):
    #    return 1.0 * grad

    #@tf.RegisterGradient("CustomGrad50")
    #def _const_mul_grad(unused_op, grad):
    #    return 1.0 * grad

    def _build_model(self):
        # Input and visual encoding layers
        self.inputs_unit_type = tf.placeholder(shape=[None, self.INPUT_IMAGE, self.INPUT_IMAGE], dtype=tf.float32,
                                               name="inputs_unit_type")
        self.inputs_workers = tf.placeholder(shape=[None, self.INPUT_IMAGE, self.INPUT_IMAGE], dtype=tf.float32,
                                               name="inputs_unit_type")

        self.input_player = tf.placeholder(shape=[None, self.NUM_SINGLE], dtype=tf.float32, name="input_player")

        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "image_unit_type: ")
        # image_selected_type = tf.Print(image_selected_type, [image_selected_type], "image_selected_type: ")

        type_flatten = self.get_flatten_conv(self.inputs_unit_type)
        workers_flatten = self.get_flatten_conv(self.inputs_workers)
        type_flatten_value = self.get_flatten_conv(self.inputs_unit_type)
        workers_flatten_value = self.get_flatten_conv(self.inputs_workers)


        flatten = tf.concat([type_flatten, self.input_player, workers_flatten], axis=1)
        flatten_value = tf.concat([type_flatten_value, self.input_player, workers_flatten_value], axis=1)

        hidden1 = slim.fully_connected(flatten, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))
        hidden2 = slim.fully_connected(hidden1, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))

        value_hidden1 = slim.fully_connected(flatten_value, 1, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))
        value_hidden2 = slim.fully_connected(value_hidden1, 1, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))


        self.batchsize = tf.placeholder(tf.int32, None, name='a')

        # batchsize = 2
        D_in, D_out = 1000, 256

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(300, activation=LeakyReLU())
        self.state_in = lstm_cell.zero_state(tf.shape(hidden2)[0], tf.float32)

        rnn_out, self.state_out = lstm_cell(hidden2, self.state_in)
        rnn_v = value_hidden2
        #rnn_out = hidden2
        hidden_out = slim.fully_connected(rnn_out, 100, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))
        hidden_out = hidden2


        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(hidden_out, self.NUM_ACTIONS,
                                           activation_fn=None,
                                           weights_initializer=normalized_columns_initializer(0.01),
                                           weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                           biases_initializer=None)
        self.available_actions = tf.placeholder(tf.float32, shape=(self.NUM_ACTIONS), name="x_t")
        # self.a_sample = self.categorical_sample(self.policy, self.NUM_ACTIONS)[0, :]
        # self.x_sample = self.categorical_sample(self.policy_x, self.NUM_COORDS_X)[0, :]
        # self.y_sample = self.categorical_sample(self.policy_y, self.NUM_COORDS_Y)[0, :]


        self.policy_x_select_point = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)
        self.policy_y_select_point = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_x_spawningPool = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          biases_initializer=None)

        self.policy_y_spawningPool = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_x_spineCrawler = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_y_spineCrawler = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_x_Gather = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                    activation_fn=None,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                    biases_initializer=None)

        self.policy_y_Gather = slim.fully_connected(slim.fully_connected(hidden_out, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)), 84,
                                                    activation_fn=None,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                    biases_initializer=None)

        self.value = slim.fully_connected(hidden_out, 1,
                                          activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(1.0),
                                          biases_initializer=None)

    def _build_graph(self):
        self.a_policy_target =  tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS), name="a_policy_target")

        self.x_t_select_point = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_select_point")
        self.y_t_select_point = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_Y), name="y_t_select_point")

        self.x_t_spawningPool = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_spawningPool")
        self.y_t_spawningPool = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_Y), name="y_t_spawningPool")

        self.x_t_spineCrawler = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_spineCrawler")
        self.y_t_spineCrawler = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_Y), name="y_t_spineCrawler")

        self.x_t_Gather = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_Gather")
        self.y_t_Gather = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_Y), name="y_t_Gather")

        self.a_t = tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS), name="a_t")
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1),
                                  name="r_t")  # not immediate, but discounted n step reward
        self.v_t = tf.placeholder(tf.float32, shape=(None, 1),
                                  name="v_t")  # not immediate, but discounted n step reward
        self.v_toggle = tf.placeholder(tf.float32, shape=(None, 1),
                                  name="v_t")  # not immediate, but discounted n step reward

        self.class_weight = tf.placeholder(tf.float32, shape=(self.NUM_ACTIONS), name="class_weight")

        self.action_weight = tf.placeholder(tf.float32, 1)
        self.value_weight = tf.placeholder(tf.float32, 1)

        self.first_v_loss = tf.placeholder(tf.float32, 1)
        self.first_a_loss = tf.placeholder(tf.float32, 1)
        self.first_x_select_loss = tf.placeholder(tf.float32, 1)
        self.first_y_select_loss = tf.placeholder(tf.float32, 1)
        self.first_x_spawn_loss = tf.placeholder(tf.float32, 1)
        self.first_y_spawn_loss = tf.placeholder(tf.float32, 1)
        self.first_x_spine_loss = tf.placeholder(tf.float32, 1)
        self.first_y_spine_loss = tf.placeholder(tf.float32, 1)

        self.first_v_loss = tf.Print(self.first_v_loss, [self.first_v_loss, tf.shape(self.first_v_loss)], "self.first_v_loss: ")
        self.first_a_loss = tf.Print(self.first_a_loss, [self.first_a_loss, tf.shape(self.first_a_loss)], "self.first_a_loss: ")
        self.first_x_select_loss = tf.Print(self.first_x_select_loss, [self.first_x_select_loss, tf.shape(self.first_x_select_loss)], "self.first_x_select_loss")
        self.first_y_select_loss = tf.Print(self.first_y_select_loss, [self.first_y_select_loss, tf.shape(self.first_y_select_loss)], "self.first_y_select_loss: ")
        self.first_x_spawn_loss = tf.Print(self.first_x_spawn_loss, [self.first_x_spawn_loss, tf.shape(self.first_x_spawn_loss)], "self.first_x_spawn_loss: ")
        self.first_y_spawn_loss = tf.Print(self.first_y_spawn_loss, [self.first_y_spawn_loss, tf.shape(self.first_y_spawn_loss)], "self.first_y_spawn_loss: ")
        self.first_x_spine_loss = tf.Print(self.first_x_spine_loss, [self.first_x_spine_loss, tf.shape(self.first_x_spine_loss)], "self.first_x_spine_loss: ")
        self.first_y_spine_loss = tf.Print(self.first_y_spine_loss, [self.first_y_spine_loss, tf.shape(self.first_y_spine_loss)], "self.first_y_spine_loss: ")

        advantage = (self.value - self.r_t) * 10
        advantage = tf.squeeze(advantage)
        advantage = tf.square(advantage) * tf.sign(advantage)
        advantage = tf.Print(advantage, [advantage, tf.shape(advantage)], "advantage: ")

        a_log_soft = tf.nn.log_softmax(self.policy)
        a_log_soft = tf.Print(a_log_soft, [a_log_soft, tf.shape(a_log_soft)], "a_log_soft: ")

        a_soft = tf.nn.softmax(self.policy)

        #a_log_prob = -tf.reduce_sum(self.class_weight * self.a_t * tf.sign(self.policy) * self.policy/tf.stop_gradient(self.policy), axis=1)
        a_log_prob = -tf.reduce_sum(a_log_soft * self.a_t * self.class_weight, axis=1)

        self.x_loss_select_point = self.get_loss_one(advantage, self.policy_x_select_point, self.x_t_select_point) * 1000
        self.y_loss_select_point = self.get_loss_one(advantage, self.policy_y_select_point, self.y_t_select_point) * 1000

        self.x_loss_spawningPool = self.get_loss_one(advantage, self.policy_x_spawningPool, self.x_t_spawningPool) * 10000
        self.y_loss_spawningPool = self.get_loss_one(advantage, self.policy_y_spawningPool, self.y_t_spawningPool) * 10000

        self.x_loss_spineCrawler = self.get_loss_one(advantage, self.policy_x_spineCrawler, self.x_t_spineCrawler) * 100000
        self.y_loss_spineCrawler = self.get_loss_one(advantage, self.policy_y_spineCrawler, self.y_t_spineCrawler) * 100000

        self.x_loss_Gather = self.get_loss_one(advantage, self.policy_x_Gather, self.x_t_Gather) * 10
        self.y_loss_Gather = self.get_loss_one(advantage, self.policy_y_Gather, self.y_t_Gather) * 10

        # self.x_loss = tf.Print(self.x_loss, [self.x_loss, tf.shape(self.x_loss)], "self.x_loss: ")

        y_log_soft_select_point = tf.nn.log_softmax(self.policy_y_select_point)
        y_log_prob_select_point = tf.reduce_sum(y_log_soft_select_point * self.y_t_select_point, axis=1)

        # x_log_prob = tf.log(tf.reduce_sum(self.policy_x * self.x_t, axis=1) + 1e-10)
        # y_log_prob = tf.log(tf.reduce_sum(self.policy_y * self.y_t, axis=1) + 1e-10)

        a_log_prob = tf.Print(a_log_prob, [a_log_prob, tf.shape(a_log_prob)], "a_log_prob: ")

        g = tf.get_default_graph()
        # with g.gradient_override_map({"Identity": "CustomGrad5"}):
        a_loss_policy = a_log_prob * tf.stop_gradient(advantage) # maximize policy
        target_error =  tf.square(
            self.policy - tf.stop_gradient(10 *  self.a_t * tf.expand_dims(tf.stop_gradient(tf.sign(-advantage)), -1)))
        target_error = tf.Print(target_error, [target_error, tf.shape(target_error)], "target_error: ")

        target_loss = tf.reduce_mean(target_error * self.a_t, axis=1)
        target_loss = tf.Print(target_loss, [target_loss, tf.shape(target_loss)], "target_loss: ")

        self.a_loss_policy = (target_loss * tf.stop_gradient(tf.abs(advantage)))


        a_policy_entropy = tf.reduce_mean(tf.reduce_mean(np.square(self.policy))) * 0.0001
        a_policy_entropy = tf.Print(a_policy_entropy, [a_policy_entropy, tf.shape(a_policy_entropy)], "a_policy_entropy: ")


        loss_value = tf.abs((self.value - self.r_t))  # minimize value error
        self.reduced_adv = tf.reduce_mean(advantage)
        self.reduced_adv = tf.Print(self.reduced_adv, [self.reduced_adv, tf.shape(self.reduced_adv)],
                                    "self.reduced_adv: ")


        prob_tf = tf.nn.softmax(self.policy)
        mean_prob = tf.reduce_mean(prob_tf, axis=0)
        mean_prob = tf.Print(mean_prob, [mean_prob, tf.shape(mean_prob)], "mean_prob: ")

        a_entropy = - tf.reduce_mean(mean_prob * tf.log(mean_prob))
        a_entropy = tf.Print(a_entropy, [a_entropy, tf.shape(a_entropy)], "a_entropy: ")

        # a_entropy =  tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        # x_entropy =  tf.reduce_sum(self.policy_x * tf.log(self.policy_x + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        # y_entropy =  tf.reduce_sum(self.policy_y * tf.log(self.policy_y + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)


        self.v_loss = loss_value * 10
        self.v_loss = tf.Print(self.v_loss, [self.v_loss, tf.shape(self.v_loss)], "self.v_loss: ")

        optimizer = tf.train.AdamOptimizer(1e-5)
        #optimizer = tf.train.GradientDescentOptimizer(1e-3)

        #gradients, variables = zip(*optimizer.compute_gradients(tf.reduce_mean(tf.reduce_mean(np.square(self.policy), axis=1)) * 0.001 +
        #                                                         self.action_weight * (self.a_loss )
        #                                                        + self.action_weight * self.x_loss_select_point + self.action_weight * self.y_loss_select_point
        #                                                        + self.action_weight * self.x_loss_spawningPool + self.action_weight * self.y_loss_spawningPool
        #                                                        + self.action_weight * self.x_loss_spineCrawler + self.action_weight * self.y_loss_spineCrawler
        #                                                        + self.action_weight * self.x_loss_Gather + self.action_weight * self.y_loss_Gather
        #                                                       ))

        self.a_loss_policy = tf.Print(self.a_loss_policy, [self.a_loss_policy, tf.shape(self.a_loss_policy)], "self.a_loss_policy: ")
        self.x_loss_select_point = tf.Print(self.x_loss_select_point, [self.x_loss_select_point, tf.shape(self.x_loss_select_point)], "self.x_loss_select_point: ")
        self.y_loss_select_point = tf.Print(self.y_loss_select_point, [self.y_loss_select_point, tf.shape(self.y_loss_select_point)], "self.y_loss_select_point")

        self.x_loss_spawningPool = tf.Print(self.x_loss_spawningPool, [self.x_loss_spawningPool, tf.shape(self.x_loss_spawningPool)], "self.x_loss_spawningPool")
        self.y_loss_spawningPool = tf.Print(self.y_loss_spawningPool, [self.y_loss_spawningPool, tf.shape(self.y_loss_spawningPool)], "self.y_loss_spawningPool")

        self.x_loss_spineCrawler = tf.Print(self.x_loss_spineCrawler, [self.x_loss_spineCrawler, tf.shape(self.x_loss_spineCrawler)], "self.x_loss_spineCrawler")
        self.y_loss_spineCrawler = tf.Print(self.y_loss_spineCrawler, [self.y_loss_spineCrawler, tf.shape(self.y_loss_spineCrawler)], "self.y_loss_spineCrawler")

        self.x_loss_Gather = tf.Print(self.x_loss_Gather, [self.x_loss_Gather, tf.shape(self.x_loss_Gather)], "self.x_loss_Gather")
        self.y_loss_Gather = tf.Print(self.y_loss_Gather, [self.y_loss_Gather, tf.shape(self.y_loss_Gather)], "self.y_loss_Gather")

        self.total_loss = self.v_loss/tf.stop_gradient(self.first_v_loss) + self.a_loss_policy/tf.stop_gradient(self.first_a_loss) +\
                          self.x_loss_select_point/tf.stop_gradient(self.first_x_select_loss) + self.y_loss_select_point/tf.stop_gradient(self.first_y_select_loss) +\
                          self.x_loss_spawningPool/tf.stop_gradient(self.first_x_spawn_loss) + self.y_loss_spawningPool/tf.stop_gradient(self.first_y_spawn_loss) +\
                          self.x_loss_spineCrawler/tf.stop_gradient(self.first_x_spine_loss) + self.y_loss_spineCrawler/tf.stop_gradient(self.first_y_spine_loss) +\
                          self.x_loss_Gather + self.y_loss_Gather
        self.minimize = optimizer.minimize(tf.reduce_mean(self.total_loss)
                                           )
        #gradients, _ = tf.clip_by_global_norm(gradients, 10.0)

        #self.minimize = optimizer.apply_gradients(zip(gradients, variables))



        #gradients_value, variables_value = zip(*optimizer.compute_gradients(self.v_loss))
        #gradients_value, _ = tf.clip_by_global_norm(gradients_value, 10.0)

        optimizer_value = tf.train.AdamOptimizer(1e-3)
        self.minimize_value = optimizer_value.minimize(self.v_loss)
        #self.minimize_value = optimizer.apply_gradients(zip(gradients_value, variables_value))

        # optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)

        # a_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)
        # x_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)
        # y_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)

        # v_optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        # v_optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE)


        # a_minimize = a_optimizer.minimize(a_loss)
        # x_minimize = x_optimizer.minimize(x_loss)
        # y_minimize = y_optimizer.minimize(y_loss)
        # v_minimize = v_optimizer.minimize(v_loss)


        # grads = tf.gradients(self.a_loss + self.x_loss + self.y_loss + self.v_loss, var_list)
        # gvs = v_optimizer.compute_gradients(self.a_loss + self.x_loss + self.y_loss + self.v_loss)

        # capped_gvs, _ = tf.clip_by_global_norm(gvs, 40.0)
        # self.minimize = v_optimizer.apply_gradients(capped_gvs)

        # self.minimize = v_optimizer.minimize(self.minimize)
        # self.minimize = v_optimizer.minimize(self.v_loss)

    def get_loss_one(self, advantage, policy, t):
        #a_log_prob = -tf.reduce_sum(t * tf.stop_gradient(tf.sign(policy)) * policy/tf.stop_gradient(policy), axis=1)
        a_log_soft = tf.nn.log_softmax(policy)
        a_log_prob = -tf.reduce_sum(a_log_soft * t, axis=1)

        a_loss_policy = a_log_prob * tf.stop_gradient(advantage)
        #entropy
        prob_tf = tf.nn.softmax(policy)
        mean_prob = tf.reduce_mean(prob_tf, axis=0)
        mean_prob = tf.Print(mean_prob, [mean_prob, tf.shape(mean_prob)], "mean_prob: ")

        a_entropy = - tf.reduce_mean(mean_prob * tf.log(mean_prob))
        value_regulizer = tf.reduce_mean(tf.reduce_mean(np.square(policy), axis=1)) * 0.000

        t_count = tf.stop_gradient(1 + tf.reduce_sum(t))
        t_count = tf.Print(t_count, [t_count, tf.shape(t_count)], "t_count: ")

        loss = tf.reduce_mean(
            tf.square(policy - (10 * t * tf.expand_dims(tf.stop_gradient(tf.sign(-advantage)), -1)))
            * t, axis=1) * tf.stop_gradient(tf.abs(advantage)) / t_count

        return loss * 10 + value_regulizer

        #return -tf.reduce_mean(a_loss_policy) + value_regulizer

    def get_loss(self, advantage, policy, t):
        x_log_soft_select_point = tf.nn.log_softmax(policy)
        x_log_prob_select_point = tf.reduce_sum(x_log_soft_select_point * t, axis=1)
        x_loss_policy_select_point = - x_log_prob_select_point * tf.stop_gradient(advantage)

        #entropy
        log_prob_tf = tf.nn.log_softmax(policy)
        prob_tf = tf.nn.softmax(policy)

        return -tf.reduce_mean(x_loss_policy_select_point) + - tf.reduce_mean(prob_tf * log_prob_tf) * self.LOSS_ENTROPY



    def categorical_sample(self, logits, d):
        self.weights = logits - tf.reduce_max(logits, [1], keep_dims=True)
        value = tf.squeeze(tf.multinomial(self.weights, 1), [1])
        return tf.one_hot(value, d)

    def train(self, a, r, v, v_toggle, s, rnn_state, class_weights, a_policy, losses):

        _, v_loss, total_loss, a_loss, x_loss, y_loss, y_loss_spawning, y_loss_spine, reduced_adv = self.session.run(
            [self.minimize, self.v_loss, self.total_loss, self.a_loss_policy, self.x_loss_select_point, self.y_loss_select_point, self.y_loss_spawningPool, self.y_loss_spineCrawler, self.reduced_adv],
            feed_dict={
                       self.inputs_unit_type: s[0],
                       self.inputs_workers: s[2],
                       self.input_player: s[1],
                       self.a_t: a[0],
                       self.x_t_select_point: a[1],
                       self.y_t_select_point: a[2],
                       self.x_t_spawningPool: a[3],
                       self.y_t_spawningPool: a[4],
                       self.x_t_spineCrawler: a[5],
                       self.y_t_spineCrawler: a[6],
                       self.x_t_Gather: a[7],
                       self.y_t_Gather: a[8],
                       self.r_t: r,
                       self.v_t:v,
                       self.v_toggle: v_toggle,
                       self.state_in[0]: rnn_state[0],
                       self.state_in[1]: rnn_state[1],
                       self.class_weight: class_weights,
                       self.action_weight: [1],
                       self.value_weight: [.00000],
                       self.a_policy_target: a_policy,
                       self.first_v_loss:np.array([np.mean(losses[0]) + 0.1]),
                       self.first_a_loss:np.array([np.mean(losses[1]) + 0.1]),
                       self.first_x_select_loss:np.array([np.mean(losses[2]) + 0.1]),
                       self.first_y_select_loss:np.array([np.mean(losses[3]) + 0.1]),
                       self.first_x_spawn_loss:np.array([np.mean(losses[4]) + 0.1]),
                       self.first_y_spawn_loss:np.array([np.mean(losses[5]) + 0.1]),
                       self.first_x_spine_loss:np.array([np.mean(losses[6]) + 0.1]),
                       self.first_y_spine_loss:np.array([np.mean(losses[7]) + 0.1]),
                       })
        return total_loss, v_loss, a_loss, x_loss, y_loss, y_loss_spawning, y_loss_spine

    def get_losses(self, a, r, v, v_toggle, s, rnn_state, class_weights, a_policy):

        losses = self.session.run(
            [
             self.v_loss, self.a_loss_policy,
             self.x_loss_select_point, self.y_loss_select_point,
             self.x_loss_spawningPool, self.y_loss_spawningPool,
             self.x_loss_spineCrawler, self.y_loss_spineCrawler,
             ],
            feed_dict={
                       self.inputs_unit_type: s[0],
                       self.inputs_workers: s[2],
                       self.input_player: s[1],
                       self.a_t: a[0],
                       self.x_t_select_point: a[1],
                       self.y_t_select_point: a[2],
                       self.x_t_spawningPool: a[3],
                       self.y_t_spawningPool: a[4],
                       self.x_t_spineCrawler: a[5],
                       self.y_t_spineCrawler: a[6],
                       self.x_t_Gather: a[7],
                       self.y_t_Gather: a[8],
                       self.r_t: r,
                       self.v_t:v,
                       self.v_toggle: v_toggle,
                       self.state_in[0]: rnn_state[0],
                       self.state_in[1]: rnn_state[1],
                       self.class_weight: class_weights,
                       self.action_weight: [1],
                       self.value_weight: [.00000],
                       self.a_policy_target: a_policy
                       })
        return losses

    def train_value(self, a, r, v, v_toggle, s, rnn_state, class_weights):

        _, v_loss, reduced_adv = self.session.run([self.minimize_value, self.v_loss, self.reduced_adv],
                                                  feed_dict={self.inputs_unit_type: s[0],
                                                             self.input_player: s[1],
                                                             self.inputs_workers: s[2],
                                                             self.a_t: a[0],
                                                             self.x_t_select_point: a[1],
                                                             self.y_t_select_point: a[2],
                                                             self.x_t_spawningPool: a[3],
                                                             self.y_t_spawningPool: a[4],
                                                             self.x_t_spineCrawler: a[5],
                                                             self.y_t_spineCrawler: a[6],
                                                             self.x_t_Gather: a[7],
                                                             self.y_t_Gather: a[8],
                                                             self.r_t: r,
                                                             self.v_t: v,
                                                             self.v_toggle: v_toggle,
                                                             self.state_in[0]: rnn_state[0],
                                                             self.state_in[1]: rnn_state[1],
                                                             self.class_weight: class_weights,
                                                             self.action_weight: [0.0],
                                                             self.value_weight: [1]
                                                             })
        return v_loss

    def save(self):
        self.saver.save(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def restore(self):
        self.saver.restore(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def predict(self, available_actions, s, batch_rnn_state):
        with self.default_graph.as_default():
            state_out, policy, policy_x_select_point, policy_y_select_point, policy_x_spawningPool, policy_y_spawningPool, policy_x_spineCrawler, policy_y_spineCrawler, policy_x_Gather, policy_y_Gather, v, batch_rnn_state = \
                self.session.run([
                                  self.state_out,
                                  self.policy,
                                  self.policy_x_select_point,
                                  self.policy_y_select_point,
                                  self.policy_x_spawningPool,
                                  self.policy_y_spawningPool,
                                  self.policy_x_spineCrawler,
                                  self.policy_y_spineCrawler,
                                  self.policy_x_Gather,
                                  self.policy_y_Gather,
                                  self.value,
                                  self.state_out],
                                 feed_dict={
                                     self.available_actions: available_actions,
                                     self.inputs_unit_type: s[0],
                                     self.inputs_workers: s[2],
                                     self.input_player: s[1],
                                     self.state_in[0]: batch_rnn_state[0],
                                     self.state_in[1]: batch_rnn_state[1]})

            if random.random() > 0.99:
                print("value: ", v)
                print("policy: ", policy)
                #print("policy_x: ", x_select_point)
                #print("policy_y: ", y_select_point)
                print("x: ", policy_x_select_point)
                print("policy_y_spawningPool: ", policy_y_spawningPool)
                print("policy_y_spineCrawler: ", policy_y_spineCrawler)
            for i in range(len(policy)):
                policy[i][2] = np.mean([max(policy_x_select_point[i]), max(policy_y_select_point[i])])
                policy[i][6] = np.mean([max(policy_x_spawningPool[i]), max(policy_y_spawningPool[i])])
                policy[i][9] = np.mean([max(policy_x_spineCrawler[i]), max(policy_y_spineCrawler[i])])
                policy[i][10] = np.mean([max(policy_x_Gather[i]), max(policy_y_Gather[i])])

            a = [self.normalized_multinomial(available_actions, p, 10) for p in np.copy(policy)]
            x_select_point = [self.normalized_multinomial(1, p, 10000) for p in np.copy(policy_x_select_point)]
            y_select_point = [self.normalized_multinomial(1, p, 10000) for p in policy_y_select_point]
            x_spawningPool = [self.normalized_multinomial(1, p, 10000) for p in policy_x_spawningPool]
            y_spawningPool = [self.normalized_multinomial(1, p, 10000) for p in policy_y_spawningPool]
            x_spineCrawler = [self.normalized_multinomial(1, p, 10000) for p in policy_x_spineCrawler]
            y_spineCrawler = [self.normalized_multinomial(1, p, 10000) for p in policy_y_spineCrawler]
            x_Gather = [self.normalized_multinomial(1, p, 10000) for p in policy_x_Gather]
            y_Gather = [self.normalized_multinomial(1, p, 10000) for p in policy_y_Gather]


            return a, x_select_point, y_select_point, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather, v, state_out, policy[0]

    def normalized_multinomial(self, available_actions, policy, n=1):
        policy[np.where(np.array(available_actions) != 1)] = max(policy)
        policy = policy - min(policy)
        policy = policy * available_actions
        if sum(policy) == 0:
            return 0
        policy = normalize(policy[:, np.newaxis], axis=0).ravel()
        multinomial = np.random.multinomial(n, policy / sum(policy) - .0000001, size=1)

        return np.argmax(multinomial)

    def get_flatten_conv(self, image_unit_type):
        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "get_flatten_conv: ")

        type_conv1 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=image_unit_type, num_outputs=128,
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 kernel_size=4, stride=2, padding='SAME')
        # type_conv1 = tf.Print(type_conv1, [type_conv1], "type_conv1: ")

        type_conv2 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=type_conv1, num_outputs=128,
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 kernel_size=4, stride=2, padding='SAME')

        type_conv3 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=type_conv2, num_outputs=128,
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 kernel_size=4, stride=2, padding='SAME')

        type_conv4 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=type_conv3, num_outputs=128,
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 kernel_size=4, stride=2, padding='SAME')

        type_conv5 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=type_conv4, num_outputs=128,
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 kernel_size=4, stride=2, padding='SAME')

        type_conv6 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=type_conv5, num_outputs=128,
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 kernel_size=4, stride=2, padding='SAME')

        # type_conv2 = tf.Print(type_conv2, [type_conv2], "type_conv2: ")

        type_flatten = slim.flatten(type_conv6)
        return type_flatten


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer
