import random

import tensorflow as tf
from sklearn.preprocessing import normalize

from tensorflow.contrib import slim

import numpy as np
from tensorflow.python.training.optimizer import _OptimizableVariable


class Network:
    NUM_ACTIONS = 15
    NUM_COORDS_X = 1764

    INPUT_IMAGE = 84
    NUM_SINGLE = 22

    LEARNING_RATE = 1e-7
    LOSS_V = .1  # v loss coefficient
    LOSS_ENTROPY = .001  # entropy coefficient
    WEIGHT_DECAY = 0.01
    PRINT_DEBUG = False

    def __init__(self):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        np.set_printoptions(precision=3)

        self._build_model()
        self._build_graph()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.session.run(tf.global_variables_initializer())
        self.restore()

        self.default_graph = tf.get_default_graph()


    def _build_model(self):
        # Input and visual encoding layers
        self.inputs_unit_type = tf.placeholder(shape=[None, self.INPUT_IMAGE, self.INPUT_IMAGE], dtype=tf.float32,
                                               name="inputs_unit_type")
        self.future_inputs_unit_type = tf.placeholder(shape=[None, self.INPUT_IMAGE, self.INPUT_IMAGE],
                                                      dtype=tf.float32,
                                                      name="inputs_unit_type")
        self.inputs_workers = tf.placeholder(shape=[None, self.INPUT_IMAGE, self.INPUT_IMAGE], dtype=tf.float32,
                                             name="inputs_unit_type")

        self.input_player = tf.placeholder(shape=[None, self.NUM_SINGLE], dtype=tf.float32, name="input_player")

        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "image_unit_type: ")
        # image_selected_type = tf.Print(image_selected_type, [image_selected_type], "image_selected_type: ")

        type_flatten = self.get_flatten_conv(self.inputs_unit_type)
        workers_flatten = self.get_flatten_conv(self.inputs_workers)

        value_hidden2 = self.get_network()

        flatten = tf.concat([type_flatten, self.input_player, workers_flatten], axis=1)
        hidden1 = slim.fully_connected(flatten, 1000, activation_fn=tf.nn.leaky_relu)
        hidden2 = slim.fully_connected(hidden1, 1000, activation_fn=tf.nn.leaky_relu)

        self.batchsize = tf.placeholder(tf.int32, None, name='a')

        # batchsize = 2
        D_in, D_out = 1000, 256

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(300, activation=tf.nn.leaky_relu)
        self.state_in = lstm_cell.zero_state(tf.shape(hidden2)[0], tf.float32)

        rnn_out, self.state_out = lstm_cell(hidden2, self.state_in)
        rnn_v = value_hidden2
        # rnn_out = hidden2
        hidden_out = slim.fully_connected(rnn_out, 100, activation_fn=tf.nn.leaky_relu,
                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))
        hidden_out = hidden2
        hidden_out = tf.nn.dropout(hidden_out, 1)  # DROP-OUT here

        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(self.get_network(), self.NUM_ACTIONS,
                                           activation_fn=None,
                                           weights_initializer=normalized_columns_initializer(0.01),
                                           weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                           )
        self.available_actions = tf.placeholder(tf.float32, shape=(self.NUM_ACTIONS), name="x_t")
        # self.a_sample = self.categorical_sample(self.policy, self.NUM_ACTIONS)[0, :]
        # self.x_sample = self.categorical_sample(self.policy_x, self.NUM_COORDS_X)[0, :]
        # self.y_sample = self.categorical_sample(self.policy_y, self.NUM_COORDS_Y)[0, :]

        self.policy_x_select_point = self.get_conv_out(1764, 42)

        self.policy_x_spawningPool = self.get_conv_out(1764, 42)

        self.policy_x_spineCrawler = self.get_conv_out(1764, 42)

        self.policy_x_Gather = self.get_conv_out(1764, 42)
        self.policy_x_extractor = self.get_conv_out(1764, 42)
        self.policy_minimap_attack = self.get_conv_out(1024, 32)
        self.policy_minimap_move = self.get_conv_out(1024, 32)
        self.policy_hatchery = self.get_conv_out(1764, 42)

        self.value = slim.fully_connected(self.get_network(),
                                          1,
                                          activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(1.0),
                                          weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                          biases_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                          )

        #self.predicted_map = slim.conv2d(activation_fn=tf.nn.leaky_relu,
        #                                 inputs=self.future_inputs_unit_type, num_outputs=1,
        #                                 kernel_size=16, stride=1)

    def get_conv_out(self, outvalues, shape):
        connected = slim.fully_connected(self.get_network(), outvalues, activation_fn=tf.nn.leaky_relu,
                                         weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY))
        out_pick = tf.reshape(connected, [-1, shape, shape, 1])

        type_conv2 = tf.layers.conv2d(
            activation=tf.nn.leaky_relu,
            inputs=out_pick,
            filters=1,
            kernel_size=[5, 5],
            padding='SAME',
        )

        # type_conv2 = tf.Print(type_conv2, [type_conv2, tf.shape(type_conv2)], "type_conv2: ")

        flatten_pick = slim.flatten(type_conv2)
        return flatten_pick

    def get_network(self):
        type_flatten_value = self.get_flatten_conv(self.inputs_unit_type)
        # type_flatten_value = tf.layers.batch_normalization(type_flatten_value)
        type_flatten_value = self.gaussian_noise_layer(type_flatten_value, 0)

        workers_flatten_value = self.get_flatten_conv(self.inputs_workers)
        # workers_flatten_value = tf.layers.batch_normalization(workers_flatten_value)
        workers_flatten_value = self.gaussian_noise_layer(workers_flatten_value, 0)

        flatten_value = tf.concat([type_flatten_value, self.input_player, workers_flatten_value], axis=1)
        value_hidden1 = slim.fully_connected(flatten_value, 100, activation_fn=tf.nn.leaky_relu,
                                             weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                             )

        value_hidden2 = tf.concat([(self.input_player), (value_hidden1)], axis=1)

        value_hidden3 = slim.fully_connected(value_hidden2, 100, activation_fn=tf.nn.leaky_relu,
                                             weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                             )
        # value_hidden3 = tf.layers.batch_normalization(value_hidden3)

        value_hidden4 = tf.nn.dropout(value_hidden3, .9)
        return value_hidden4

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def _build_graph(self):
        self.a_policy_target = tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS), name="a_policy_target")

        self.x_t_select_point = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_select_point")

        self.x_t_spawningPool = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_spawningPool")

        self.x_t_spineCrawler = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_spineCrawler")
        self.x_t_Gather = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_Gather")
        self.x_t_extractor = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_extractor")
        self.x_t_minimap_attack = tf.placeholder(tf.float32, shape=(None, 1024), name="x_t_minimap_attack")
        self.x_t_minimap_move = tf.placeholder(tf.float32, shape=(None, 1024), name="x_t_minimap_move")
        self.x_t_hatchery = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t_hatchery")

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
        self.unknown_weights = tf.placeholder(tf.float32, shape=(None))
        if self.PRINT_DEBUG:
            self.unknown_weights = tf.Print(self.unknown_weights,
                                            [self.unknown_weights, tf.shape(self.unknown_weights)],
                                            "self.unknown_weights: ")
        self.future_v = tf.placeholder(tf.float32, shape=(None, 1),
                                       name="future_v")

        self.first_v_loss = tf.placeholder(tf.float32, 1)
        self.first_a_loss = tf.placeholder(tf.float32, 12)
        self.first_x_select_loss = tf.placeholder(tf.float32, 1)
        self.first_x_spawn_loss = tf.placeholder(tf.float32, 1)
        self.first_x_spine_loss = tf.placeholder(tf.float32, 1)
        if self.PRINT_DEBUG:
            self.first_v_loss = tf.Print(self.first_v_loss, [self.first_v_loss, tf.shape(self.first_v_loss)],
                                         "self.first_v_loss: ")
        if self.PRINT_DEBUG:
            self.first_a_loss = tf.Print(self.first_a_loss, [self.first_a_loss, tf.shape(self.first_a_loss)],
                                         "self.first_a_loss: ", name="self.first_a_loss")
        if self.PRINT_DEBUG:
            self.first_x_select_loss = tf.Print(self.first_x_select_loss,
                                                [self.first_x_select_loss, tf.shape(self.first_x_select_loss)],
                                                "self.first_x_select_loss")
        if self.PRINT_DEBUG:
            self.first_x_spawn_loss = tf.Print(self.first_x_spawn_loss,
                                               [self.first_x_spawn_loss, tf.shape(self.first_x_spawn_loss)],
                                               "self.first_x_spawn_loss: ")
        if self.PRINT_DEBUG:
            self.first_x_spine_loss = tf.Print(self.first_x_spine_loss,
                                               [self.first_x_spine_loss, tf.shape(self.first_x_spine_loss)],
                                               "self.first_x_spine_loss: ")

        new_advantage = .5 * tf.nn.relu(self.value - self.future_v)
        new_advantage = .5 * (self.value - self.future_v)
        new_advantage = tf.Print(new_advantage,
                                 [new_advantage, tf.shape(new_advantage), tf.reduce_min(new_advantage),
                                  tf.reduce_max(new_advantage)],
                                 "new_advantage: ")

        base_advantage = self.future_v - self.value

        base_advantage = tf.Print(base_advantage,
                                  [base_advantage, tf.shape(base_advantage), tf.reduce_min(base_advantage),
                                   tf.reduce_max(base_advantage)],
                                  "base_advantage: ")
        advantage = (base_advantage) * 10
        advantage = tf.squeeze(advantage)  # + 1 * tf.sign(advantage)
        advantage = advantage + .000 * tf.sign(advantage)
        # advantage = tf.square(advantage) * tf.sign(advantage)
        # advantage = tf.nn.softmax(tf.abs(advantage)) * tf.sign(advantage) * 100

        if self.PRINT_DEBUG:
            advantage = tf.Print(advantage,
                                 [advantage, tf.shape(advantage), tf.reduce_min(advantage), tf.reduce_max(advantage)],
                                 "advantage: ")

        a_log_soft = tf.nn.log_softmax(self.policy)
        if self.PRINT_DEBUG:
            a_log_soft = tf.Print(a_log_soft, [a_log_soft, tf.shape(a_log_soft)], "a_log_soft: ")

        a_soft = tf.nn.softmax(self.policy)

        # a_log_prob = -tf.reduce_sum(self.class_weight * self.a_t * tf.sign(self.policy) * self.policy/tf.stop_gradient(self.policy), axis=1)
        a_log_prob = -tf.reduce_sum(a_log_soft * self.a_t * self.class_weight, axis=1) * 100

        self.x_loss_select_point = self.get_loss_one(advantage, self.policy_x_select_point,
                                                     self.x_t_select_point) * 100000
        self.x_loss_spawningPool = self.get_loss_one(advantage, self.policy_x_spawningPool,
                                                     self.x_t_spawningPool) * 1000000
        self.x_loss_spineCrawler = self.get_loss_one(advantage, self.policy_x_spineCrawler,
                                                     self.x_t_spineCrawler) * 1000000
        self.x_loss_extractor = self.get_loss_one(advantage, self.policy_x_extractor, self.x_t_extractor) * 1000000
        self.x_loss_minimap_attack = self.get_loss_one(advantage, self.policy_minimap_attack,
                                                       self.x_t_minimap_attack) * 1000000
        self.x_loss_Gather = self.get_loss_one(advantage, self.policy_x_Gather, self.x_t_Gather) * 1000000
        self.x_loss_minimap_move = self.get_loss_one(advantage, self.policy_minimap_move,
                                                     self.x_t_minimap_move) * 1000000
        self.x_loss_hatchery = self.get_loss_one(advantage, self.policy_hatchery, self.x_t_hatchery) * 1000000

        if self.PRINT_DEBUG:
            a_log_prob = tf.Print(a_log_prob, [a_log_prob, tf.shape(a_log_prob)], "a_log_prob: ")

        g = tf.get_default_graph()
        # with g.gradient_override_map({"Identity": "CustomGrad5"}):
        a_loss_policy = a_log_prob * tf.stop_gradient(advantage)  # maximize policy
        target_error = tf.square(
            self.policy - tf.stop_gradient(1 * self.a_t * tf.expand_dims(tf.stop_gradient(tf.sign(advantage)), -1)))
        target_error = tf.stop_gradient(tf.transpose(tf.abs([advantage]))) * target_error * self.a_t

        if self.PRINT_DEBUG:
            target_error = tf.Print(target_error, [target_error, tf.shape(target_error), tf.reduce_min(target_error),
                                                   tf.reduce_max(target_error)], "target_error: ")

        self.target_loss = tf.reduce_mean(target_error * self.a_t, axis=0)
        if self.PRINT_DEBUG:
            self.target_loss = tf.Print(self.target_loss, [self.target_loss, tf.shape(self.target_loss)],
                                        "self.target_loss: ")

        self.a_loss_policy = tf.reduce_mean(target_error, axis=1) * 1000

        # loss = tf.reduce_mean(
        #    tf.square(policy - (1 * t * tf.expand_dims(tf.stop_gradient(tf.sign(-advantage)), -1)))
        #    * t, axis=1)

        loss_value = tf.abs((self.value - self.r_t))  # minimize value error
        loss_value = tf.losses.mean_squared_error(self.r_t, self.value)
        self.reduced_adv = tf.reduce_mean(advantage)
        if self.PRINT_DEBUG:
            self.reduced_adv = tf.Print(self.reduced_adv, [self.reduced_adv, tf.shape(self.reduced_adv)],
                                        "self.reduced_adv: ")

        prob_tf = tf.nn.softmax(self.policy)
        mean_prob = tf.reduce_mean(prob_tf, axis=0)
        if self.PRINT_DEBUG:
            mean_prob = tf.Print(mean_prob, [mean_prob, tf.shape(mean_prob)], "mean_prob: ")


        self.v_loss = tf.squeeze(loss_value)  # * tf.squeeze(self.unknown_weights)
        if self.PRINT_DEBUG:
            self.v_loss = tf.Print(self.v_loss, [self.v_loss, tf.shape(self.v_loss)], "self.v_loss: ")
        # 1.032e+00 -9.617e-01
        optimizer = tf.train.AdamOptimizer(1e-6)
        predict_map_optimizer = tf.train.AdamOptimizer(1e-5)
        # optimizer = tf.train.GradientDescentOptimizer(1e-3)


        if self.PRINT_DEBUG:
            self.a_loss_policy = tf.Print(self.a_loss_policy, [self.a_loss_policy, tf.shape(self.a_loss_policy)],
                                          "self.a_loss_policy: ", name="a_loss_print")
        if self.PRINT_DEBUG:
            self.x_loss_select_point = tf.Print(self.x_loss_select_point,
                                                [self.x_loss_select_point, tf.shape(self.x_loss_select_point)],
                                                "self.x_loss_select_point: ")
        if self.PRINT_DEBUG:
            self.x_loss_spawningPool = tf.Print(self.x_loss_spawningPool,
                                                [self.x_loss_spawningPool, tf.shape(self.x_loss_spawningPool)],
                                                "self.x_loss_spawningPool")
        if self.PRINT_DEBUG:
            self.x_loss_spineCrawler = tf.Print(self.x_loss_spineCrawler,
                                                [self.x_loss_spineCrawler, tf.shape(self.x_loss_spineCrawler)],
                                                "self.x_loss_spineCrawler")
        if self.PRINT_DEBUG:
            self.x_loss_extractor = tf.Print(self.x_loss_extractor,
                                             [self.x_loss_extractor, tf.shape(self.x_loss_extractor)],
                                             "self.x_loss_extractor")
        if self.PRINT_DEBUG:
            self.x_loss_Gather = tf.Print(self.x_loss_Gather, [self.x_loss_Gather, tf.shape(self.x_loss_Gather)],
                                          "self.x_loss_Gather")

        self.a_loss_policy = self.a_loss_policy
        self.x_loss_select_point = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_select_point
        self.x_loss_spawningPool = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_spawningPool
        self.x_loss_spineCrawler = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_spineCrawler
        self.x_loss_extractor = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_extractor
        self.x_loss_Gather = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_Gather
        self.x_loss_minimap_attack = tf.stop_gradient(tf.abs(advantage)) * self.x_loss_minimap_attack

        final_v_loss = tf.reduce_mean(self.v_loss) / tf.stop_gradient(tf.reduce_mean(self.v_loss) + 0.00000001)
        if self.PRINT_DEBUG:
            final_v_loss = tf.Print(final_v_loss, [final_v_loss, tf.shape(final_v_loss)], "final_v_loss")

        final_a_loss = self.a_loss_policy / tf.stop_gradient(self.a_loss_policy + 0.00000001)
        if self.PRINT_DEBUG:
            final_a_loss = tf.Print(final_a_loss, [final_a_loss, tf.shape(final_a_loss)], "final_a_loss")

        final_select_loss = tf.reduce_mean(self.x_loss_select_point) / tf.stop_gradient(
            tf.reduce_mean(self.x_loss_select_point) + 0.00000001)
        if self.PRINT_DEBUG:
            final_select_loss = tf.Print(final_select_loss, [final_select_loss, tf.shape(final_select_loss)],
                                         "final_select_loss")

        final_spawn_loss = tf.reduce_mean(self.x_loss_spawningPool) / (
            tf.stop_gradient(tf.reduce_mean(self.x_loss_spawningPool) + 0.00000001))
        if self.PRINT_DEBUG:
            final_spawn_loss = tf.Print(final_spawn_loss, [final_spawn_loss, tf.shape(final_spawn_loss)],
                                        "final_spawn_loss")

        final_spine_loss = tf.reduce_mean(self.x_loss_spineCrawler) / (
            tf.stop_gradient(tf.reduce_mean(self.x_loss_spineCrawler) + 0.00000001))
        if self.PRINT_DEBUG:
            final_spine_loss = tf.Print(final_spine_loss, [final_spine_loss, tf.shape(final_spine_loss)],
                                        "final_spine_loss")

        final_extractor_loss = tf.reduce_mean(self.x_loss_extractor) / (
            tf.stop_gradient(tf.reduce_mean(self.x_loss_extractor) + 0.00000001))
        if self.PRINT_DEBUG:
            final_extractor_loss = tf.Print(final_extractor_loss,
                                            [final_extractor_loss, tf.shape(final_extractor_loss)],
                                            "final_extractor_loss")

        final_gather_loss = tf.reduce_mean(self.x_loss_Gather) / tf.stop_gradient(
            tf.reduce_mean((self.x_loss_Gather) + 0.00000001))
        if self.PRINT_DEBUG:
            final_gather_loss = tf.Print(final_gather_loss, [final_gather_loss, tf.shape(final_gather_loss)],
                                         "final_gather_loss")

        select_positive_loss = tf.reduce_mean(tf.reduce_mean(1 - self.policy_x_select_point, axis=1))
        spawn_positive_loss = tf.reduce_mean(tf.reduce_mean(1 - self.policy_x_spawningPool, axis=1))
        spine_positive_loss = tf.reduce_mean(tf.reduce_mean(1 - self.policy_x_spineCrawler, axis=1))
        self.gather_positive_loss = tf.reduce_mean(tf.square(1 - self.policy_x_Gather))
        self.extractor_positive_loss = tf.reduce_mean(tf.square(1 - self.policy_x_extractor))
        # positive_loss = 0.000000 * (positive_loss)
        # positive_loss = tf.Print(positive_loss, [positive_loss, tf.shape(positive_loss)], "positive_loss")

        x_loss_select_point_entropy = 10000 * tf.reduce_sum(
            self.x_loss_select_point * tf.log(self.x_loss_select_point + 1e-10))
        x_loss_extractor_entropy = 10000 * tf.reduce_sum(self.x_loss_extractor * tf.log(self.x_loss_extractor + 1e-10))
        x_loss_Gather_entropy = 10000 * tf.reduce_sum(self.x_loss_Gather * tf.log(self.x_loss_Gather + 1e-10))

        self.gather_positive_loss = 10 * tf.Print(self.gather_positive_loss,
                                                  [self.gather_positive_loss, tf.shape(self.gather_positive_loss)],
                                                  "self.gather_positive_loss")
        self.extractor_positive_loss = 10 * tf.Print(self.extractor_positive_loss, [self.extractor_positive_loss,
                                                                                    tf.shape(
                                                                                        self.extractor_positive_loss)],
                                                     "self.extractor_positive_loss")
        select_positive_loss = 10 * tf.Print(select_positive_loss,
                                             [select_positive_loss, tf.shape(select_positive_loss)],
                                             "select_positive_loss")
        negative_loss = \
            tf.reduce_mean(tf.reduce_mean(tf.abs(self.policy_x_select_point))) \
            + tf.reduce_mean(tf.reduce_mean(tf.abs(self.policy_x_spawningPool))) \
            + tf.reduce_mean(tf.reduce_mean(tf.abs(self.policy_x_spineCrawler))) \
            + tf.reduce_mean(tf.reduce_mean(tf.abs(self.policy_x_Gather))) \
            + tf.reduce_mean(tf.reduce_mean(tf.abs(self.policy_x_extractor)))

        # positive_loss = 10 * positive_loss
        # positive_loss = tf.Print(positive_loss, [positive_loss, tf.shape(positive_loss)], "positive_loss")

        self.total_loss = final_v_loss + \
                          tf.reduce_sum(final_a_loss) + \
                          final_select_loss + \
                          final_spawn_loss + \
                          final_spine_loss + \
                          final_extractor_loss + \
                          final_gather_loss  # + \
        # (positive_loss)

        self.minimize_value = predict_map_optimizer.minimize(self.v_loss + .000 * tf.reduce_sum(tf.abs(self.value)))
        self.minimize_a = optimizer.minimize(
            tf.reduce_mean(self.a_loss_policy) + 0.000 * tf.reduce_sum(tf.abs(self.a_loss_policy)))
        self.minimize_select = optimizer.minimize(
            tf.reduce_mean(self.x_loss_select_point) + 0 * select_positive_loss + 0 * x_loss_select_point_entropy)
        self.minimize_spawn = optimizer.minimize(
            tf.reduce_mean(self.x_loss_spawningPool) + 0.000 * tf.reduce_sum(tf.abs(self.x_loss_spawningPool)))
        self.minimize_spine = optimizer.minimize(
            tf.reduce_mean(self.x_loss_spineCrawler) + 0.000 * tf.reduce_sum(tf.abs(self.x_loss_spineCrawler)))
        self.minimize_extract = optimizer.minimize(
            tf.reduce_mean(self.x_loss_extractor) + 0 * self.extractor_positive_loss + 0 * x_loss_extractor_entropy)
        self.minimize_gather = optimizer.minimize(
            tf.reduce_mean(self.x_loss_Gather) + 0 * self.gather_positive_loss + 0 * x_loss_Gather_entropy)
        self.minimize_minimap_attack = optimizer.minimize(tf.reduce_mean(self.x_loss_minimap_attack))
        self.minimize_minimap_move = optimizer.minimize(tf.reduce_mean(self.x_loss_minimap_move))
        self.minimize_hatchery = optimizer.minimize(tf.reduce_mean(self.x_loss_hatchery))

        #self.map_predict_loss = tf.reduce_sum(tf.square(self.future_inputs_unit_type - self.predicted_map))
        #self.minimize_predict_map = predict_map_optimizer.minimize(self.map_predict_loss)


    def get_loss_one(self, advantage, policy, t):
        prob_tf = tf.nn.softmax(policy)
        mean_prob = tf.reduce_mean(prob_tf, axis=0)
        mean_prob = tf.Print(mean_prob, [mean_prob, tf.shape(mean_prob)], "mean_prob: ")

        t_count = tf.stop_gradient(1 + tf.reduce_sum(t))
        t_count = tf.Print(t_count, [t_count, tf.shape(t_count)], "t_count: ")

        loss = tf.reduce_mean(
            tf.square(policy - (1 * t * tf.expand_dims(tf.stop_gradient(tf.sign(advantage)), -1)))
            * t, axis=1)
        p_max = tf.argmax(policy, axis=1)
        t_max = tf.argmax(t, axis=1)
        positive = advantage > 0
        actions_of_choose = tf.not_equal(p_max, t_max)
        bit_loss = tf.stop_gradient(tf.logical_or(tf.logical_and(actions_of_choose, tf.logical_not(positive)),
                                                  tf.logical_and(tf.logical_not(actions_of_choose), positive)));

        bit_loss = tf.Print(bit_loss, [bit_loss, tf.shape(t_count)], "bit_loss: ")
        # use estimated value
        return loss  # * self.unknown_weights #+ self.unknown_weights/100000

        # return -tf.reduce_mean(a_loss_policy) + value_regulizer

    def get_loss(self, advantage, policy, t):
        x_log_soft_select_point = tf.nn.log_softmax(policy)
        x_log_prob_select_point = tf.reduce_sum(x_log_soft_select_point * t, axis=1)
        x_loss_policy_select_point = - x_log_prob_select_point * tf.stop_gradient(advantage)

        # entropy
        log_prob_tf = tf.nn.log_softmax(policy)
        prob_tf = tf.nn.softmax(policy)

        return -tf.reduce_mean(x_loss_policy_select_point) + - tf.reduce_mean(prob_tf * log_prob_tf) * self.LOSS_ENTROPY

    def categorical_sample(self, logits, d):
        self.weights = logits - tf.reduce_max(logits, [1], keep_dims=True)
        value = tf.squeeze(tf.multinomial(self.weights, 1), [1])
        return tf.one_hot(value, d)

    def train_value(self, data, future_v):

        _, v_loss = self.session.run(
            [
                self.minimize_value,
                self.v_loss,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.r_t: data[9],
                self.v_t: data[13],
            })
        return v_loss

    def train_a(self, data, future_v):

        _, loss = self.session.run(
            [
                self.minimize_a,
                self.a_loss_policy,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.a_t: data[0],
            })
        return loss

    def train_select(self, data, future_v):

        _, loss = self.session.run(
            [
                self.minimize_select,
                self.x_loss_select_point,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_select_point: data[1],
            })
        return loss


    def train_spawn(self, data, future_v):

        _, loss = self.session.run(
            [
                self.minimize_spawn,
                self.x_loss_spawningPool,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_spawningPool: data[2],
            })
        return loss

    def train_spine(self, data, future_v):

        _, loss = self.session.run(
            [
                self.minimize_spine,
                self.x_loss_spineCrawler,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_spineCrawler: data[3],
            })
        return loss


    def train_extractor(self, data, future_v):

        _, loss = self.session.run(
            [
                self.minimize_extract,
                self.x_loss_extractor,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_extractor: data[5],
            })
        return loss

    def train_gather(self, data, future_v):

        _, loss = self.session.run(
            [
                self.minimize_gather,
                self.x_loss_Gather,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_Gather: data[4],
            })
        return loss

    def train_minimap_attack(self, data, future_v):

        _, = self.session.run(
            [
                self.minimize_minimap_attack,

            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_minimap_attack: data[6],
            })

    def train_minimap_move(self, data, future_v):

        _, = self.session.run(
            [
                self.minimize_minimap_move,

            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_minimap_move: data[7],
            })

    def train_hatchery(self, data, future_v):

        _, = self.session.run(
            [
                self.minimize_hatchery,
            ]
            ,
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.x_t_hatchery: data[8],
            })


    def train(self, data, future_v):
        _, _, _, _, _, _, _, _, _, _, v_loss, total_loss, a_loss, x_loss, x_loss_spawning, x_loss_spine, x_loss_gather, x_loss_extractor, reduced_adv = self.session.run(
            [
                self.minimize_value,
                self.minimize_a,
                self.minimize_select,
                self.minimize_spawn,
                self.minimize_spine,
                self.minimize_extract,
                self.minimize_gather,
                self.minimize_minimap_attack,
                self.minimize_minimap_move,
                self.minimize_hatchery,
                self.v_loss,
                self.total_loss,
                self.a_loss_policy,
                self.x_loss_select_point,
                self.x_loss_spawningPool,
                self.x_loss_spineCrawler,
                self.x_loss_Gather,
                self.x_loss_extractor,
                self.reduced_adv],
            feed_dict={
                self.inputs_unit_type: data[10],
                self.input_player: data[11],
                self.inputs_workers: data[12],
                self.future_v: future_v,

                self.a_t: data[0],
                self.x_t_select_point: data[1],
                self.x_t_spawningPool: data[2],
                self.x_t_spineCrawler: data[3],
                self.x_t_Gather: data[4],
                self.x_t_extractor: data[5],
                self.x_t_minimap_attack: data[6],
                self.x_t_minimap_move: data[7],
                self.x_t_hatchery: data[8],
                self.r_t: data[9],
                self.v_t: data[13],
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
                self.v_t: v,
                # self.v_toggle: v_toggle,
                #                       self.state_in[0]: rnn_state[0],
                #                       self.state_in[1]: rnn_state[1],
                #                       self.class_weight: class_weights,
                self.action_weight: [1],
                self.value_weight: [.00000],
            })
        return losses

    def save(self):
        self.saver.save(self.session, 'brain/models/model-' + str(1) + '.cptk')
        pass

    def restore(self):
        self.saver.restore(self.session, 'brain/models/model-' + str(1) + '.cptk')
        pass

    def predict_value(self, available_actions, s, batch_rnn_state):
        with self.default_graph.as_default():
            v = \
                self.session.run([
                    self.value
                ],
                    feed_dict={
                        self.available_actions: available_actions,
                        self.inputs_unit_type: s[0],
                        self.inputs_workers: s[2],
                        self.input_player: s[1]
                        # self.state_in[0]: batch_rnn_state[0],
                        # self.state_in[1]: batch_rnn_state[1]
                    })
            return v[0]

    def train_predict_map(self, available_actions, s, batch_rnn_state):
        with self.default_graph.as_default():
            map_predict_loss, _ = \
                self.session.run([
                    self.map_predict_loss,
                    self.minimize_predict_map
                ],
                    feed_dict={
                        self.available_actions: available_actions,
                        self.future_inputs_unit_type: s[0],
                        self.inputs_workers: s[2],
                        self.input_player: s[1]
                        # self.state_in[0]: batch_rnn_state[0],
                        # self.state_in[1]: batch_rnn_state[1]
                    })
            return map_predict_loss

    def predict(self, available_actions, s, batch_rnn_state):
        with self.default_graph.as_default():
            state_out, policy, policy_x_select_point, policy_x_spawningPool, policy_x_spineCrawler, policy_x_Gather, policy_x_extractor, policy_minimap_attack, policy_minimap_move, policy_hatchery, v, batch_rnn_state = \
                self.session.run([
                    self.state_out,
                    self.policy,
                    self.policy_x_select_point,
                    self.policy_x_spawningPool,
                    self.policy_x_spineCrawler,
                    self.policy_x_Gather,
                    self.policy_x_extractor,
                    self.policy_minimap_attack,
                    self.policy_minimap_move,
                    self.policy_hatchery,
                    self.value,
                    self.state_out],
                    feed_dict={
                        self.available_actions: available_actions,
                        self.inputs_unit_type: s[0],
                        self.inputs_workers: s[2],
                        self.input_player: s[1]
                        # self.state_in[0]: batch_rnn_state[0],
                        # self.state_in[1]: batch_rnn_state[1]
                    })

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
                policy[i][12] = max(policy_minimap_attack[i])
                policy[i][13] = max(policy_minimap_move[i])
                policy[i][14] = max(policy_hatchery[i])
            if r_value > 0.99:
                print("after policy: ", policy)

            a = [self.normalized_multinomial(available_actions, p, 100000) for p in np.copy(policy)]
            x_select_point = [self.normalized_multinomial(1, p, 100000000) for p in np.copy(policy_x_select_point)]
            x_spawningPool = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_spawningPool]
            x_spineCrawler = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_spineCrawler]
            x_Gather = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_Gather]
            x_extractor = [self.normalized_multinomial(1, p, 100000000) for p in policy_x_extractor]
            minimap_attack = [self.normalized_multinomial(1, p, 100000000) for p in policy_minimap_attack]
            minimap_move = [self.normalized_multinomial(1, p, 100000000) for p in policy_minimap_move]
            x_hatchery = [self.normalized_multinomial(1, p, 100000000) for p in policy_hatchery]

            return a, x_select_point, x_spawningPool, x_spineCrawler, x_Gather, x_extractor, minimap_attack, minimap_move, x_hatchery, v, state_out, \
                   policy[0]

    def normalized_multinomial(self, available_actions, policy, n=1):
        policy[np.where(np.array(available_actions) != 1)] = max(policy)
        policy = policy - min(policy)
        policy = policy * available_actions
        if sum(policy) == 0:
            return 0
        # policy = normalize(policy[:, np.newaxis], axis=0).ravel()
        return np.argmax(policy)

        # return np.argmax(multinomial)

    def get_flatten_conv(self, image_unit_type):
        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "get_flatten_conv: ")

        type_conv6 = self.get_conv(image_unit_type)

        # type_conv2 = tf.Print(type_conv2, [type_conv2], "type_conv2: ")

        type_flatten = slim.flatten(type_conv6)
        return type_flatten

    def get_conv(self, image_unit_type):


        image_unit_type = tf.expand_dims(image_unit_type, 3)

        type_conv1 = slim.conv2d(activation_fn=tf.nn.leaky_relu,
                                 inputs=image_unit_type, num_outputs=3,
                                 kernel_size=4, stride=2, padding='SAME',
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 )

        type_conv2 = slim.conv2d(activation_fn=tf.nn.leaky_relu,
                                 inputs=type_conv1, num_outputs=3,
                                 kernel_size=4, stride=2, padding='SAME',
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 )
        type_conv3 = slim.conv2d(activation_fn=tf.nn.leaky_relu,
                                 inputs=type_conv2, num_outputs=3,
                                 kernel_size=4, stride=2, padding='SAME',
                                 weights_regularizer=slim.l2_regularizer(self.WEIGHT_DECAY),
                                 )

        #type_conv3 = tf.Print(type_conv3, [type_conv3, tf.shape(type_conv3)],
        #                             "self.type_conv3: ", name="self.type_conv3")
        return type_conv3


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer
