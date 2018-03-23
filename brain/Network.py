import random

import tensorflow as tf
from sklearn.preprocessing import normalize

from tensorflow.contrib import slim

import numpy as np
from tensorflow.python.keras._impl.keras.layers import LeakyReLU
from tensorflow.python.training.optimizer import _OptimizableVariable


class Network:
    NUM_ACTIONS = 12
    NUM_COORDS_X = 1764

    INPUT_IMAGE = 84
    NUM_SINGLE = 18

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
        hidden_out = tf.nn.dropout(hidden_out, .8)  # DROP-OUT here


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


        self.policy_x_select_point = slim.fully_connected(slim.fully_connected(hidden_out, 10,
                                                                               activation_fn=LeakyReLU(),
                                                                               weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)),
                                                          1764,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)

        self.policy_x_spawningPool = slim.fully_connected(slim.fully_connected(hidden_out, 10,
                                                                               activation_fn=LeakyReLU(),
                                                                               weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)),
                                                          1764,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          biases_initializer=None)

        self.policy_x_spineCrawler = slim.fully_connected(slim.fully_connected(hidden_out, 10,
                                                                               activation_fn=LeakyReLU(),
                                                                               weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)),
                                                          1764,
                                                          activation_fn=None,
                                                          weights_initializer=normalized_columns_initializer(0.01),
                                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                          biases_initializer=None)


        self.policy_x_Gather = slim.fully_connected(slim.fully_connected(hidden_out, 10,
                                                                         activation_fn=LeakyReLU(),
                                                                         weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)),
                                                    1764,
                                                    activation_fn=None,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                                    biases_initializer=None)

        self.policy_x_extractor = slim.fully_connected(slim.fully_connected(hidden_out, 10,
                                                                         activation_fn=LeakyReLU(),
                                                                         weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)),
                                                       1764,
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

        self.x_t_spawningPool = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_spawningPool")

        self.x_t_spineCrawler = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_spineCrawler")
        self.x_t_Gather = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_Gather")
        self.x_t_extractor = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_spineCrawler")

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
        self.first_a_loss = tf.placeholder(tf.float32, 12)
        self.first_x_select_loss = tf.placeholder(tf.float32, 1)
        self.first_x_spawn_loss = tf.placeholder(tf.float32, 1)
        self.first_x_spine_loss = tf.placeholder(tf.float32, 1)

        self.first_v_loss = tf.Print(self.first_v_loss, [self.first_v_loss, tf.shape(self.first_v_loss)], "self.first_v_loss: ")
        self.first_a_loss = tf.Print(self.first_a_loss, [self.first_a_loss, tf.shape(self.first_a_loss)], "self.first_a_loss: ", name="self.first_a_loss")
        self.first_x_select_loss = tf.Print(self.first_x_select_loss, [self.first_x_select_loss, tf.shape(self.first_x_select_loss)], "self.first_x_select_loss")
        self.first_x_spawn_loss = tf.Print(self.first_x_spawn_loss, [self.first_x_spawn_loss, tf.shape(self.first_x_spawn_loss)], "self.first_x_spawn_loss: ")
        self.first_x_spine_loss = tf.Print(self.first_x_spine_loss, [self.first_x_spine_loss, tf.shape(self.first_x_spine_loss)], "self.first_x_spine_loss: ")

        advantage = (self.value - self.r_t) * 100
        advantage = tf.squeeze(advantage) #+ 1 * tf.sign(advantage)
        advantage = advantage + .000 * tf.sign(advantage)
        #advantage = tf.square(advantage) * tf.sign(advantage)
        advantage = tf.Print(advantage, [advantage, tf.shape(advantage)], "advantage: ")

        a_log_soft = tf.nn.log_softmax(self.policy)
        a_log_soft = tf.Print(a_log_soft, [a_log_soft, tf.shape(a_log_soft)], "a_log_soft: ")

        a_soft = tf.nn.softmax(self.policy)

        #a_log_prob = -tf.reduce_sum(self.class_weight * self.a_t * tf.sign(self.policy) * self.policy/tf.stop_gradient(self.policy), axis=1)
        a_log_prob = -tf.reduce_sum(a_log_soft * self.a_t * self.class_weight, axis=1) * 100

        self.x_loss_select_point = self.get_loss_one(advantage, self.policy_x_select_point, self.x_t_select_point) * 100000
        self.x_loss_spawningPool = self.get_loss_one(advantage, self.policy_x_spawningPool, self.x_t_spawningPool) * 1000000
        self.x_loss_spineCrawler = self.get_loss_one(advantage, self.policy_x_spineCrawler, self.x_t_spineCrawler) * 1000000
        self.x_loss_extractor = self.get_loss_one(advantage, self.policy_x_extractor, self.x_t_extractor) * 1000000
        self.x_loss_Gather = self.get_loss_one(advantage, self.policy_x_Gather, self.x_t_Gather) * 1000000



        a_log_prob = tf.Print(a_log_prob, [a_log_prob, tf.shape(a_log_prob)], "a_log_prob: ")

        g = tf.get_default_graph()
        # with g.gradient_override_map({"Identity": "CustomGrad5"}):
        a_loss_policy = a_log_prob * tf.stop_gradient(advantage) # maximize policy
        target_error =  tf.square(
            self.policy - tf.stop_gradient(1 *  self.a_t * tf.expand_dims(tf.stop_gradient(tf.sign(-advantage)), -1)))
        target_error = tf.Print(target_error, [target_error, tf.shape(target_error)], "target_error: ")

        self.target_loss = tf.reduce_mean(target_error * self.a_t, axis=0)
        self.target_loss = tf.Print(self.target_loss, [self.target_loss, tf.shape(self.target_loss)], "self.target_loss: ")

        self.a_loss_policy = tf.reduce_mean(tf.stop_gradient(tf.transpose(tf.abs([advantage]))) * target_error * self.a_t, axis=0) * 1000


        a_policy_entropy = tf.reduce_mean(tf.reduce_mean(np.square(self.policy))) * 0.0001
        a_policy_entropy = tf.Print(a_policy_entropy, [a_policy_entropy, tf.shape(a_policy_entropy)], "a_policy_entropy: ")


        loss_value = tf.square((self.value - self.r_t))  # minimize value error
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


        self.v_loss = loss_value
        self.v_loss = tf.Print(self.v_loss, [self.v_loss, tf.shape(self.v_loss)], "self.v_loss: ")

        optimizer = tf.train.AdamOptimizer(1e-7)
        #optimizer = tf.train.GradientDescentOptimizer(1e-3)

        #gradients, variables = zip(*optimizer.compute_gradients(tf.reduce_mean(tf.reduce_mean(np.square(self.policy), axis=1)) * 0.001 +
        #                                                         self.action_weight * (self.a_loss )
        #                                                        + self.action_weight * self.x_loss_select_point + self.action_weight * self.y_loss_select_point
        #                                                        + self.action_weight * self.x_loss_spawningPool + self.action_weight * self.y_loss_spawningPool
        #                                                        + self.action_weight * self.x_loss_spineCrawler + self.action_weight * self.y_loss_spineCrawler
        #                                                        + self.action_weight * self.x_loss_Gather + self.action_weight * self.y_loss_Gather
        #                                                       ))

        self.a_loss_policy = tf.Print(self.a_loss_policy, [self.a_loss_policy, tf.shape(self.a_loss_policy)], "self.a_loss_policy: ", name="a_loss_print")
        self.x_loss_select_point = tf.Print(self.x_loss_select_point, [self.x_loss_select_point, tf.shape(self.x_loss_select_point)], "self.x_loss_select_point: ")
        self.x_loss_spawningPool = tf.Print(self.x_loss_spawningPool, [self.x_loss_spawningPool, tf.shape(self.x_loss_spawningPool)], "self.x_loss_spawningPool")
        self.x_loss_spineCrawler = tf.Print(self.x_loss_spineCrawler, [self.x_loss_spineCrawler, tf.shape(self.x_loss_spineCrawler)], "self.x_loss_spineCrawler")
        self.x_loss_extractor = tf.Print(self.x_loss_extractor, [self.x_loss_extractor, tf.shape(self.x_loss_extractor)], "self.x_loss_extractor")
        self.x_loss_Gather = tf.Print(self.x_loss_Gather, [self.x_loss_Gather, tf.shape(self.x_loss_Gather)], "self.x_loss_Gather")

        self.a_loss_policy =  self.a_loss_policy
        self.x_loss_select_point = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_select_point
        self.x_loss_spawningPool = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_spawningPool
        self.x_loss_spineCrawler = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_spineCrawler
        self.x_loss_extractor = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_extractor
        self.x_loss_Gather = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_Gather

        final_v_loss = 10 * tf.reduce_mean(self.v_loss) / tf.stop_gradient(tf.reduce_mean(self.v_loss) + 0.00000001)
        final_v_loss = tf.Print(final_v_loss, [final_v_loss, tf.shape(final_v_loss)], "final_v_loss")

        final_a_loss = self.a_loss_policy/tf.stop_gradient(self.a_loss_policy+ 0.00000001)
        final_a_loss = tf.Print(final_a_loss, [final_a_loss, tf.shape(final_a_loss)], "final_a_loss")

        final_select_loss = tf.reduce_mean(self.x_loss_select_point) / tf.stop_gradient(tf.reduce_mean(self.x_loss_select_point) + 0.00000001)
        final_select_loss = tf.Print(final_select_loss, [final_select_loss, tf.shape(final_select_loss)], "final_select_loss")

        final_spawn_loss = tf.reduce_mean(self.x_loss_spawningPool) / (tf.stop_gradient(tf.reduce_mean(self.x_loss_spawningPool)+ 0.00000001))
        final_spawn_loss = tf.Print(final_spawn_loss, [final_spawn_loss, tf.shape(final_spawn_loss)], "final_spawn_loss")

        final_spine_loss = tf.reduce_mean(self.x_loss_spineCrawler) / (tf.stop_gradient(tf.reduce_mean(self.x_loss_spineCrawler)+ 0.00000001))
        final_spine_loss = tf.Print(final_spine_loss, [final_spine_loss, tf.shape(final_spine_loss)], "final_spine_loss")

        final_extractor_loss = tf.reduce_mean(self.x_loss_extractor) / (tf.stop_gradient(tf.reduce_mean(self.x_loss_extractor)+ 0.00000001))
        final_extractor_loss = tf.Print(final_extractor_loss, [final_extractor_loss, tf.shape(final_extractor_loss)], "final_extractor_loss")

        final_gather_loss = tf.reduce_mean(self.x_loss_Gather) / tf.stop_gradient(tf.reduce_mean((self.x_loss_Gather) + 0.00000001))
        final_gather_loss = tf.Print(final_gather_loss, [final_gather_loss, tf.shape(final_gather_loss)], "final_gather_loss")

        positive_loss =  \
            tf.reduce_mean(tf.reduce_mean(self.policy_x_select_point)) \
            + tf.reduce_mean(tf.reduce_mean(self.policy_x_spawningPool)) \
            + tf.reduce_mean(tf.reduce_mean(self.policy_x_spineCrawler)) \
            + tf.reduce_mean(tf.reduce_mean(self.policy_x_Gather)) \
            + tf.reduce_mean(tf.reduce_mean(self.policy_x_extractor))
        positive_loss = .01* positive_loss
        positive_loss = tf.Print(positive_loss, [positive_loss, tf.shape(positive_loss)], "positive_loss")

        self.total_loss = final_v_loss + \
                          tf.reduce_sum(final_a_loss) + \
                          final_select_loss + \
                          final_spawn_loss + \
                          final_spine_loss  + \
                          final_extractor_loss  + \
                          final_gather_loss + \
            - (positive_loss)
             
        self.minimize = optimizer.minimize(tf.reduce_mean(self.total_loss) #- tf.reduce_mean(tf.reduce_mean(self.policy_x_spineCrawler)/10000)
                                           )
        #gradients, _ = tf.clip_by_global_norm(gradients, 10.0)

        #self.minimize = optimizer.apply_gradients(zip(gradients, variables))



        #gradients_value, variables_value = zip(*optimizer.compute_gradients(self.v_loss))
        #gradients_value, _ = tf.clip_by_global_norm(gradients_value, 10.0)

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
            tf.square(policy - (1 * t * tf.expand_dims(tf.stop_gradient(tf.sign(-advantage)), -1)))
            * t, axis=1)

        return loss  + value_regulizer

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

    def train(self, a, r, v, v_toggle, s, rnn_state, class_weights, losses):

        _, v_loss, total_loss, a_loss, x_loss, x_loss_spawning, x_loss_spine, x_loss_gather, x_loss_extractor, reduced_adv = self.session.run(
            [self.minimize, self.v_loss, self.total_loss, self.a_loss_policy, self.x_loss_select_point, self.x_loss_spawningPool, self.x_loss_spineCrawler, self.x_loss_Gather, self.x_loss_extractor, self.reduced_adv],
            feed_dict={
                       self.inputs_unit_type: s[0],
                       self.inputs_workers: s[2],
                       self.input_player: s[1],
                       self.a_t: a[0],
                       self.x_t_select_point: a[1],
                       self.x_t_spawningPool: a[2],
                       self.x_t_spineCrawler: a[3],
                       self.x_t_Gather: a[4],
                       self.x_t_extractor: a[5],
                       self.r_t: r,
                       self.v_t:v,
                       #self.v_toggle: v_toggle,
                       #self.state_in[0]: rnn_state[0],
                       #self.state_in[1]: rnn_state[1],
 #                      self.class_weight: class_weights,
                       self.action_weight: [1],
                       self.value_weight: [.00000],
                       self.first_v_loss:np.array([np.mean(losses[0]) + 0.00001]),
                       self.first_a_loss:np.array(losses[1] + 0.001),
                       self.first_x_select_loss:np.array([np.mean(losses[2]) + 0.000001]),
                       self.first_x_spawn_loss:np.array([np.mean(losses[3]) + 0.000001]),
                       self.first_x_spine_loss:np.array([np.mean(losses[4]) + 0.000001])
                       })
        return total_loss, v_loss, a_loss, x_loss, x_loss_spawning, x_loss_spine, x_loss_gather, x_loss_extractor

    def get_losses(self, a, r, v, v_toggle, s, rnn_state, class_weights):

        losses = self.session.run(
            [
             self.v_loss, self.a_loss_policy,
             self.x_loss_select_point,
             self.x_loss_spawningPool,
             self.x_loss_spineCrawler,
             ],
            feed_dict={
                       self.inputs_unit_type: s[0],
                       self.inputs_workers: s[2],
                       self.input_player: s[1],
                       self.a_t: a[0],
                       self.x_t_select_point: a[1],
                       self.x_t_spawningPool: a[2],
                       self.x_t_spineCrawler: a[3],
                       self.x_t_Gather: a[4],
                       self.x_t_extractor: a[5],
                       self.r_t: r,
                       self.v_t:v,
                       #self.v_toggle: v_toggle,
#                       self.state_in[0]: rnn_state[0],
#                       self.state_in[1]: rnn_state[1],
#                       self.class_weight: class_weights,
                       self.action_weight: [1],
                       self.value_weight: [.00000],
                       })
        return losses



    def save(self):
        self.saver.save(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def restore(self):
        self.saver.restore(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def predict(self, available_actions, s, batch_rnn_state):
        with self.default_graph.as_default():
            state_out, policy, policy_x_select_point, policy_x_spawningPool, policy_x_spineCrawler, policy_x_Gather, policy_x_extractor, v, batch_rnn_state = \
                self.session.run([
                                  self.state_out,
                                  self.policy,
                                  self.policy_x_select_point,
                                  self.policy_x_spawningPool,
                                  self.policy_x_spineCrawler,
                                  self.policy_x_Gather,
                                  self.policy_x_extractor,
                                  self.value,
                                  self.state_out],
                                 feed_dict={
                                     self.available_actions: available_actions,
                                     self.inputs_unit_type: s[0],
                                     self.inputs_workers: s[2],
                                     self.input_player: s[1],
                                     self.state_in[0]: batch_rnn_state[0],
                                     self.state_in[1]: batch_rnn_state[1]})

            r_value = random.random()
            if r_value > 0.99:
                print("value: ", v)
                print("policy: ", policy)
                print("x: ", policy_x_select_point)
                print("policy_x_spawningPool: ", policy_x_spawningPool)
                print("policy_x_spineCrawler: ", policy_x_spineCrawler)
            for i in range(len(policy)):
                policy[i][2] = max(policy_x_select_point[i])
                policy[i][6] = max(policy_x_spawningPool[i])
                policy[i][9] = max(policy_x_spineCrawler[i])
                policy[i][10] = max(policy_x_Gather[i])
                policy[i][11] = max(policy_x_extractor[i])
            if r_value > 0.99:
                print("after policy: ", policy)

            a = [self.normalized_multinomial(available_actions, p, 100000) for p in np.copy(policy)]
            x_select_point = [self.normalized_multinomial(1, p, 100000000) for p in np.copy(policy_x_select_point)]
            x_spawningPool = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_spawningPool]
            x_spineCrawler = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_spineCrawler]
            x_Gather = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_Gather]
            x_extractor = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_extractor]


            return a, x_select_point, x_spawningPool, x_spineCrawler, x_Gather, x_extractor, v, state_out, policy[0]

    def normalized_multinomial(self, available_actions, policy, n=1):
        policy[np.where(np.array(available_actions) != 1)] = max(policy)
        policy = policy - min(policy)
        policy = policy * available_actions
        if sum(policy) == 0:
            return 0
        policy = normalize(policy[:, np.newaxis], axis=0).ravel()
        multinomial = np.random.multinomial(n, policy / sum(policy) - .000000001, size=1)

        return np.argmax(multinomial)

    def get_flatten_conv(self, image_unit_type):
        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "get_flatten_conv: ")

        type_conv1 = slim.conv2d(activation_fn=LeakyReLU(),
                                 inputs=image_unit_type, num_outputs=128,
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 kernel_size=4, stride=2, padding='SAME')
        #type_conv1 = tf.Print(type_conv1, [type_conv1], "type_conv1: ")

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

        type_flatten = slim.flatten(type_conv2)
        return type_flatten


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer
