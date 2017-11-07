import tensorflow as tf

from tensorflow.contrib import slim

import numpy as np
from tensorflow.contrib.keras.python.keras.layers import LeakyReLU


class Network:
    NUM_ACTIONS = 11
    NUM_COORDS_X = 84
    NUM_COORDS_Y = 84

    INPUT_IMAGE = 84
    NUM_SINGLE = 16

    LEARNING_RATE = 1e-7
    LOSS_V = .1  # v loss coefficient
    LOSS_ENTROPY = .01  # entropy coefficient

    def __init__(self):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self._build_model()
        self._build_graph()

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        #self.saver = tf.train.Saver(max_to_keep=5)

        self.summary_writer = tf.summary.FileWriter("train", graph=self.default_graph)

        self.default_graph.finalize()  # avoid modifications


        self.restore()

    @tf.RegisterGradient("CustomGrad5")
    def _const_mul_grad(unused_op, grad):
        return 1.0 * grad
    @tf.RegisterGradient("CustomGrad50")
    def _const_mul_grad(unused_op, grad):
        return 1.0 * grad

    def _build_model(self):
        # Input and visual encoding layers
        self.inputs_unit_type = tf.placeholder(shape=[None, self.INPUT_IMAGE, self.INPUT_IMAGE], dtype=tf.float32, name="inputs_unit_type")
        self.input_player = tf.placeholder(shape=[None, self.NUM_SINGLE], dtype=tf.float32, name="input_player")


        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "image_unit_type: ")
        # image_selected_type = tf.Print(image_selected_type, [image_selected_type], "image_selected_type: ")

        type_flatten = self.get_flatten_conv(self.inputs_unit_type)


        flatten = tf.concat([self.input_player], axis=1)

        hidden1 = slim.fully_connected(flatten, 1000, activation_fn=LeakyReLU())
        hidden2 = slim.fully_connected(hidden1, 1000, activation_fn=LeakyReLU())
        hidden3 = slim.fully_connected(hidden2, 1000, activation_fn=LeakyReLU())
        hidden4 = slim.fully_connected(hidden3, 1000, activation_fn=LeakyReLU())

        self.batchsize = tf.placeholder(tf.int32, None, name='a')

        # batchsize = 2
        D_in, D_out = 1000, 256

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        self.state_in = lstm_cell.zero_state(tf.shape(hidden4)[0], tf.float32)

        rnn_out, self.state_out = lstm_cell(hidden4, self.state_in)
        #rnn_out = hidden4

        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(rnn_out, self.NUM_ACTIONS,
                                           activation_fn=None,
                                           weights_initializer=normalized_columns_initializer(0.01),
                                           biases_initializer=None)
        self.available_actions = tf.placeholder(tf.float32, shape=(self.NUM_ACTIONS), name="x_t")
        self.a_sample = self.categorical_sample(self.policy, self.NUM_ACTIONS)[0, :]


        self.policy_x = slim.fully_connected(rnn_out, 84,
                                             activation_fn=None,
                                             weights_initializer=normalized_columns_initializer(0.01),
                                             biases_initializer=None)
        self.x_sample = self.categorical_sample(self.policy_x, self.NUM_COORDS_X)[0, :]

        self.policy_y = slim.fully_connected(rnn_out, 84,
                                             activation_fn=None,
                                             weights_initializer=normalized_columns_initializer(0.01),
                                             biases_initializer=None)
        self.y_sample = self.categorical_sample(self.policy_y, self.NUM_COORDS_Y)[0, :]

        self.value = slim.fully_connected(rnn_out, 1,
                                          activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(1.0),
                                          biases_initializer=None)

    def _build_graph(self):

        self.x_t = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t")
        self.y_t = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_Y), name="y_t")
        self.a_t = tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS), name="a_t")
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1), name="r_t")  # not immediate, but discounted n step reward

        self.class_weight = tf.placeholder(tf.float32, shape=(self.NUM_ACTIONS), name="class_weight")


        a_reduced = tf.reduce_sum(self.a_t, axis=1)
        x_reduced = tf.reduce_sum(self.x_t, axis=1)
        y_reduced = tf.reduce_sum(self.y_t, axis=1)

        t = self.policy * self.a_t
        t = tf.Print(t, [t, tf.shape(t)], "t: ")

        a_log_soft = tf.multiply(tf.nn.log_softmax(self.policy), self.class_weight)
        a_soft = tf.nn.softmax(self.policy)
        a_log_prob = tf.reduce_sum(a_log_soft * self.a_t, axis=1)


        x_log_soft = tf.nn.log_softmax(self.policy_x)
        x_log_prob = tf.reduce_sum(x_log_soft * self.x_t, axis=1)

        y_log_soft = tf.nn.log_softmax(self.policy_y)
        y_log_prob = tf.reduce_sum(y_log_soft * self.y_t, axis=1)

        #x_log_prob = tf.log(tf.reduce_sum(self.policy_x * self.x_t, axis=1) + 1e-10)
        #y_log_prob = tf.log(tf.reduce_sum(self.policy_y * self.y_t, axis=1) + 1e-10)

        a_log_prob = tf.Print(a_log_prob, [a_log_prob, tf.shape(a_log_prob)], "a_log_prob: ")


        advantage =  (self.r_t - self.value)

        advantage = tf.squeeze(advantage)
        #advantage = tf.square(advantage) * tf.sign(advantage)
        advantage = tf.Print(advantage, [advantage, tf.shape(advantage)], "advantage: ")

        g = tf.get_default_graph()

        #with g.gradient_override_map({"Identity": "CustomGrad5"}):
        a_loss_policy = tf.identity(- a_log_prob  * tf.stop_gradient(tf.square(advantage) * 100) , name="Identity")  # maximize policy

        with g.gradient_override_map({"Identity": "CustomGrad5"}):
            x_loss_policy = tf.identity((- x_log_prob * tf.stop_gradient(tf.square(advantage) * 100) ), name="Identity")  # maximize policy
        with g.gradient_override_map({"Identity": "CustomGrad5"}):
            y_loss_policy = tf.identity((- y_log_prob * tf.stop_gradient(tf.square(advantage) * 100) ), name="Identity")  # maximize policy

        with g.gradient_override_map({"Identity": "CustomGrad50"}):
            loss_value = tf.identity(tf.abs(advantage), name="Identity")    # minimize value error
        self.reduced_adv =  tf.reduce_mean(advantage)
        self.reduced_adv = tf.Print(self.reduced_adv, [self.reduced_adv, tf.shape(self.reduced_adv)], "self.reduced_adv: ")

        a_loss_policy = tf.Print(a_loss_policy, [a_loss_policy, tf.shape(a_loss_policy)], "a_loss_policy: ")

        prob_tf = tf.nn.softmax(self.policy)
        a_entropy = - tf.reduce_sum(prob_tf * a_log_soft)
        a_entropy = tf.Print(a_entropy, [a_entropy, tf.shape(a_entropy)], "a_entropy: ")

        #a_entropy =  tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        x_entropy =  tf.reduce_sum(self.policy_x * tf.log(self.policy_x + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        y_entropy =  tf.reduce_sum(self.policy_y * tf.log(self.policy_y + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)

        #loss_total = tf.reduce_mean(a_loss_policy + x_loss_policy + y_loss_policy + loss_value + entropy)
        self.a_loss = tf.reduce_mean(a_loss_policy)
        self.a_loss = tf.Print(self.a_loss, [self.a_loss, tf.shape(self.a_loss)], "self.a_loss: ")

        self.x_loss = tf.reduce_mean(x_loss_policy)
        self.x_loss = tf.Print(self.x_loss, [self.x_loss, tf.shape(self.x_loss)], "self.x_loss: ")

        self.y_loss = tf.reduce_mean(y_loss_policy)
        self.y_loss = tf.Print(self.y_loss, [self.y_loss, tf.shape(self.y_loss)], "self.y_loss: ")

        self.v_loss = tf.reduce_mean(loss_value)
        self.v_loss = tf.Print(self.v_loss, [self.v_loss, tf.shape(self.v_loss)], "self.v_loss: ")


        optimizer = tf.train.AdamOptimizer(1e-5)
        gradients, variables = zip(*optimizer.compute_gradients( self.a_loss + self.x_loss + self.y_loss + .1*self.v_loss ))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.minimize = optimizer.apply_gradients(zip(gradients, variables))

        #optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)

        #a_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)
        #x_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)
        #y_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)

        #v_optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        #v_optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE)


        #a_minimize = a_optimizer.minimize(a_loss)
        #x_minimize = x_optimizer.minimize(x_loss)
        #y_minimize = y_optimizer.minimize(y_loss)
        #v_minimize = v_optimizer.minimize(v_loss)


        #grads = tf.gradients(self.a_loss + self.x_loss + self.y_loss + self.v_loss, var_list)
        #gvs = v_optimizer.compute_gradients(self.a_loss + self.x_loss + self.y_loss + self.v_loss)

        #capped_gvs, _ = tf.clip_by_global_norm(gvs, 40.0)
        #self.minimize = v_optimizer.apply_gradients(capped_gvs)

        #self.minimize = v_optimizer.minimize(self.minimize)
        #self.minimize = v_optimizer.minimize(self.v_loss)

    def categorical_sample(self, logits, d):
        self.weights = logits - tf.reduce_max(logits, [1], keep_dims=True)
        value = tf.squeeze(tf.multinomial(self.weights, 1), [1])
        return tf.one_hot(value, d)

    def train(self, a, r, s, rnn_state, class_weights):

        _, a_loss, x_loss, y_loss, v_loss, reduced_adv  = self.session.run([self.minimize, self.a_loss, self.x_loss, self.y_loss, self.v_loss, self.reduced_adv],
                                                  feed_dict={self.inputs_unit_type: s[0],
                                                             self.input_player: s[1],
                                                             self.a_t : a[0],
                                                             self.x_t : a[1],
                                                             self.y_t : a[2],
                                                             self.r_t : r,
                                                             self.state_in[0]: rnn_state[0],
                                                             self.state_in[1]: rnn_state[1],
                                                             self.class_weight: class_weights
                                                             })
        return a_loss, x_loss, y_loss, v_loss

    def save(self):
        #self.saver.save(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def restore(self):
        #self.saver.restore(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def predict(self, available_actions, s, batch_rnn_state):
        with self.default_graph.as_default():
            weights, a, policy, policy_x, policy_y, x, y, v, batch_rnn_state  = self.session.run([self.weights, self.a_sample, self.policy, self.policy_x, self.policy_y, self.x_sample, self.y_sample, self.value, self.state_out],
                                   feed_dict={
                                       self.available_actions: available_actions,
                                       self.inputs_unit_type: s[0],
                                              self.input_player: s[1],
                                              self.state_in[0] : batch_rnn_state[0],
                                              self.state_in[1]: batch_rnn_state[1]})
            return a, x, y, v, batch_rnn_state



    def get_flatten_conv(self, image_unit_type):
        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "get_flatten_conv: ")

        type_conv1 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=image_unit_type, num_outputs=16,
                                 kernel_size=8, stride=4, padding='VALID')
        # type_conv1 = tf.Print(type_conv1, [type_conv1], "type_conv1: ")

        type_conv2 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=type_conv1, num_outputs=32,
                                 kernel_size=4, stride=2, padding='VALID')
        # type_conv2 = tf.Print(type_conv2, [type_conv2], "type_conv2: ")

        type_flatten = slim.flatten(type_conv2)
        return type_flatten

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer