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
    WEIGHT_DECAY = 0.001
    def __init__(self):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        np.set_printoptions(precision=3)

        self._build_model()
        self._build_graph()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.session.run(tf.global_variables_initializer())
        self.restore()

        self.default_graph = tf.get_default_graph()

        self.summary_writer = tf.summary.FileWriter("train", graph=self.default_graph)

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

        value_hidden1 = slim.fully_connected(flatten_value, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))
        value_hidden2 = slim.fully_connected(value_hidden1, 1000, activation_fn=LeakyReLU(), weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))


        self.batchsize = tf.placeholder(tf.int32, None, name='a')

        # batchsize = 2
        D_in, D_out = 1000, 256

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        self.state_in = lstm_cell.zero_state(tf.shape(hidden2)[0], tf.float32)

        rnn_out, self.state_out = lstm_cell(value_hidden2, self.state_in)
        rnn_v = rnn_out
        rnn_out = hidden2

        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(rnn_out, self.NUM_ACTIONS,
                                           activation_fn=None,
                                           weights_initializer=normalized_columns_initializer(0.01),
                                           weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                           biases_initializer=None)
        self.available_actions = tf.placeholder(tf.float32, shape=(self.NUM_ACTIONS), name="x_t")
        # self.a_sample = self.categorical_sample(self.policy, self.NUM_ACTIONS)[0, :]
        # self.x_sample = self.categorical_sample(self.policy_x, self.NUM_COORDS_X)[0, :]
        # self.y_sample = self.categorical_sample(self.policy_y, self.NUM_COORDS_Y)[0, :]


        self.policy_x_select_point = slim.fully_connected(rnn_out, 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)
        self.policy_y_select_point = slim.fully_connected(rnn_out, 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_x_spawningPool = slim.fully_connected(rnn_out, 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          biases_initializer=None)

        self.policy_y_spawningPool = slim.fully_connected(rnn_out, 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_x_spineCrawler = slim.fully_connected(rnn_out, 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_y_spineCrawler = slim.fully_connected(rnn_out, 84,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_x_Gather = slim.fully_connected(rnn_out, 84,
                                                    activation_fn=None,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                    biases_initializer=None)

        self.policy_y_Gather = slim.fully_connected(rnn_out, 84,
                                                    activation_fn=None,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                    biases_initializer=None)

        self.value = slim.fully_connected(rnn_v, 1,
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

        advantage = (self.value - self.r_t)
        advantage = tf.squeeze(advantage)
        #advantage = tf.square(advantage*10) * tf.sign(advantage)
        advantage = tf.Print(advantage, [advantage, tf.shape(advantage)], "advantage: ")

        a_log_soft = tf.nn.log_softmax(self.policy)
        a_log_soft = tf.Print(a_log_soft, [a_log_soft, tf.shape(a_log_soft)], "a_log_soft: ")

        a_soft = tf.nn.softmax(self.policy)

        #a_log_prob = -tf.reduce_sum(self.class_weight * self.a_t * tf.sign(self.policy) * self.policy/tf.stop_gradient(self.policy), axis=1)
        a_log_prob = -tf.reduce_sum(a_log_soft * self.a_t * self.class_weight, axis=1)

        self.x_loss_select_point = self.get_loss_one(advantage, self.policy_x_select_point, self.x_t_select_point)
        self.y_loss_select_point = self.get_loss_one(advantage, self.policy_y_select_point, self.y_t_select_point)

        self.x_loss_spawningPool = self.get_loss_one(advantage, self.policy_x_spawningPool, self.x_t_spawningPool)
        self.y_loss_spawningPool = self.get_loss_one(advantage, self.policy_y_spawningPool, self.y_t_spawningPool)

        self.x_loss_spineCrawler = self.get_loss_one(advantage, self.policy_x_spineCrawler, self.x_t_spineCrawler)
        self.y_loss_spineCrawler = self.get_loss_one(advantage, self.policy_y_spineCrawler, self.y_t_spineCrawler)

        self.x_loss_Gather = self.get_loss_one(advantage, self.policy_x_Gather, self.x_t_Gather)
        self.y_loss_Gather = self.get_loss_one(advantage, self.policy_y_Gather, self.y_t_Gather)

        # self.x_loss = tf.Print(self.x_loss, [self.x_loss, tf.shape(self.x_loss)], "self.x_loss: ")

        y_log_soft_select_point = tf.nn.log_softmax(self.policy_y_select_point)
        y_log_prob_select_point = tf.reduce_sum(y_log_soft_select_point * self.y_t_select_point, axis=1)

        # x_log_prob = tf.log(tf.reduce_sum(self.policy_x * self.x_t, axis=1) + 1e-10)
        # y_log_prob = tf.log(tf.reduce_sum(self.policy_y * self.y_t, axis=1) + 1e-10)

        a_log_prob = tf.Print(a_log_prob, [a_log_prob, tf.shape(a_log_prob)], "a_log_prob: ")

        g = tf.get_default_graph()
        # with g.gradient_override_map({"Identity": "CustomGrad5"}):
        a_loss_policy = a_log_prob * tf.stop_gradient(advantage) # maximize policy
        a_loss_policy = tf.reduce_mean(self.class_weight * tf.square(self.policy - (10 * self.a_t * tf.expand_dims(tf.stop_gradient(tf.sign(-advantage)),-1)))
                                       * self.a_t, axis=1)  * tf.stop_gradient(tf.abs(advantage))

        a_loss_policy = tf.Print(a_loss_policy, [a_loss_policy, tf.shape(a_loss_policy)], "a_loss_policy: ")
        #a_loss_policy = tf.reduce_mean(a_loss_policy, axis=0)
        #a_loss_policy = tf.Print(a_loss_policy, [a_loss_policy, tf.shape(a_loss_policy)], "a_loss_policy: ")

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

        # loss_total = tf.reduce_mean(a_loss_policy + x_loss_policy + y_loss_policy + loss_value + entropy)
        self.a_loss = -tf.reduce_mean(a_loss_policy) + -a_entropy * self.LOSS_ENTROPY
        self.a_loss = tf.Print(self.a_loss, [self.a_loss, tf.shape(self.a_loss)], "self.a_loss: ")

        self.v_loss = tf.reduce_mean(loss_value)
        self.v_loss = tf.Print(self.v_loss, [self.v_loss, tf.shape(self.v_loss)], "self.v_loss: ")

        optimizer = tf.train.AdamOptimizer(1e-4)
        #optimizer = tf.train.GradientDescentOptimizer(1e-3)

        gradients, variables = zip(*optimizer.compute_gradients(tf.reduce_mean(tf.reduce_mean(np.square(self.policy), axis=1)) * 0.001 +
                                                                 self.action_weight * (self.a_loss )
                                                                + self.action_weight * self.x_loss_select_point + self.action_weight * self.y_loss_select_point
                                                                + self.action_weight * self.x_loss_spawningPool + self.action_weight * self.y_loss_spawningPool
                                                                + self.action_weight * self.x_loss_spineCrawler + self.action_weight * self.y_loss_spineCrawler
                                                                + self.action_weight * self.x_loss_Gather + self.action_weight * self.y_loss_Gather
                                                               ))

        self.minimize = optimizer.minimize(a_loss_policy
                                           + self.x_loss_select_point
                                           + self.y_loss_select_point
                                           + self.x_loss_spawningPool
                                           + self.y_loss_spawningPool
                                           + self.x_loss_spineCrawler
                                           + self.y_loss_spineCrawler
                                           + self.x_loss_Gather
                                           + self.y_loss_Gather
                                           )
        #gradients, _ = tf.clip_by_global_norm(gradients, 10.0)

        #self.minimize = optimizer.apply_gradients(zip(gradients, variables))



        gradients_value, variables_value = zip(*optimizer.compute_gradients( self.value_weight * self.v_loss))
        gradients_value, _ = tf.clip_by_global_norm(gradients_value, 10.0)

        optimizer_value = tf.train.AdamOptimizer(1e-5)
        self.minimize_value = optimizer_value.minimize(self.value_weight * self.v_loss)
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
        value_regulizer = tf.reduce_mean(tf.reduce_mean(np.square(policy), axis=1)) * 0.001

        loss = tf.reduce_mean(
            tf.square(policy - (10 * t * tf.expand_dims(tf.stop_gradient(tf.sign(-advantage)), -1)))
            * t, axis=1) * tf.stop_gradient(tf.abs(advantage))
        return loss

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

    def train(self, a, r, v, v_toggle, s, rnn_state, class_weights, a_policy):

        _, a_loss, x_loss, y_loss, reduced_adv = self.session.run(
            [self.minimize, self.a_loss, self.x_loss_select_point, self.y_loss_select_point, self.reduced_adv],
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
        return a_loss, x_loss, y_loss

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

            a = self.normalized_multinomial(available_actions, policy[0], 1)
            x_select_point = self.normalized_multinomial(1, policy_x_select_point[0], 10)
            y_select_point = self.normalized_multinomial(1, policy_y_select_point[0], 10)
            x_spawningPool = self.normalized_multinomial(1, policy_x_spawningPool[0], 10)
            y_spawningPool = self.normalized_multinomial(1, policy_y_spawningPool[0], 10)
            x_spineCrawler = self.normalized_multinomial(1, policy_x_spineCrawler[0], 10)
            y_spineCrawler = self.normalized_multinomial(1, policy_y_spineCrawler[0], 10)
            x_Gather = self.normalized_multinomial(1, policy_x_Gather[0], 100)
            y_Gather = self.normalized_multinomial(1, policy_y_Gather[0], 100)

            if random.random() > 0.99:
                print("value: ", v)
                print("policy: ", policy)
                print("policy_x: ", x_select_point)
                print("policy_y: ", y_select_point)
                print("x: ", policy_x_select_point)
            return a, x_select_point, y_select_point, x_spawningPool, y_spawningPool, x_spineCrawler, y_spineCrawler, x_Gather, y_Gather, v, state_out, policy[0]

    def normalized_multinomial(self, available_actions, policy, n=1):
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
