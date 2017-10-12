import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *
from tensorflow.contrib import slim


class Network:
    NUM_ACTIONS = 10
    NUM_COORDS_X = 84
    NUM_COORDS_Y = 84

    INPUT_IMAGE = 84
    NUM_SINGLE = 14

    LEARNING_RATE = 5e-3
    LOSS_V = .1  # v loss coefficient
    LOSS_ENTROPY = .01  # entropy coefficient

    def __init__(self):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self._build_model()
        self._build_graph()

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        #self.saver = tf.train.Saver(max_to_keep=5)

        self.summary_writer = tf.summary.FileWriter("train", graph=self.default_graph)

        self.default_graph.finalize()  # avoid modifications


        self.restore()

    @tf.RegisterGradient("CustomGrad")
    def _const_mul_grad(unused_op, grad):
        return 5.0 * grad

    def _build_model(self):
        # Input and visual encoding layers
        self.inputs_unit_type = tf.placeholder(shape=[None, self.INPUT_IMAGE, self.INPUT_IMAGE], dtype=tf.float32, name="inputs_unit_type")
        self.input_player = tf.placeholder(shape=[None, self.NUM_SINGLE], dtype=tf.float32, name="input_player")


        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "image_unit_type: ")
        # image_selected_type = tf.Print(image_selected_type, [image_selected_type], "image_selected_type: ")

        type_flatten = self.get_flatten_conv(self.inputs_unit_type)


        flatten = tf.concat([type_flatten, self.input_player], axis=1)

        hidden1 = slim.fully_connected(flatten, 1000, activation_fn=tf.nn.elu)
        hidden2 = slim.fully_connected(hidden1, 1000, activation_fn=tf.nn.elu)
        hidden3 = slim.fully_connected(hidden2, 1000, activation_fn=tf.nn.elu)

        self.batchsize = tf.placeholder(tf.int32, None, name='a')

        # batchsize = 2
        D_in, D_out = 1000, 256

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        self.state_in = lstm_cell.zero_state(tf.shape(hidden3)[0], tf.float32)

        rnn_out, self.state_out = lstm_cell(hidden3, self.state_in)


        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(rnn_out, self.NUM_ACTIONS,
                                           activation_fn=tf.nn.softmax,
                                           weights_initializer=normalized_columns_initializer(0.01),
                                           biases_initializer=None)
        self.policy_x = slim.fully_connected(rnn_out, 84,
                                             activation_fn=tf.nn.softmax,
                                             weights_initializer=normalized_columns_initializer(0.01),
                                             biases_initializer=None)
        self.policy_y = slim.fully_connected(rnn_out, 84,
                                             activation_fn=tf.nn.softmax,
                                             weights_initializer=normalized_columns_initializer(0.01),
                                             biases_initializer=None)
        self.value = slim.fully_connected(rnn_out, 1,
                                          activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(1.0),
                                          biases_initializer=None)

    def _build_model2(self):

        image_input = Input(shape=(1, self.INPUT_IMAGE, self.INPUT_IMAGE), name="image_input")
        single_input = Input(batch_shape=(None, self.NUM_SINGLE), name="single_input")

        image_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        image_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_conv1)

        concatenated_input = keras.layers.concatenate([Flatten()(image_conv2), single_input])
        l_dense = Dense(160, activation='relu',)(concatenated_input)
        l_dense = Dense(160, activation='relu')(l_dense)
        l_dense = Dense(160, activation='relu')(l_dense)

        out_actions = Dense(self.NUM_ACTIONS, activation='softmax')(l_dense)
        out_x = Dense(self.NUM_COORDS_X, activation='softmax')(l_dense)
        out_y = Dense(self.NUM_COORDS_Y, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[image_input, single_input], outputs=[out_actions, out_x, out_y, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self):

        self.x_t = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_X), name="x_t")
        self.y_t = tf.placeholder(tf.float32, shape=(None, self.NUM_COORDS_Y), name="y_t")
        self.a_t = tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS), name="a_t")
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1), name="r_t")  # not immediate, but discounted n step reward


        a_reduced = tf.reduce_sum(self.a_t, axis=1)
        x_reduced = tf.reduce_sum(self.x_t, axis=1)
        y_reduced = tf.reduce_sum(self.y_t, axis=1)

        t = self.policy * self.a_t
        t = tf.Print(t, [t, tf.shape(t)], "t: ")

        a_log_prob = tf.log(tf.reduce_sum(t, axis=1) + 1e-10)
        x_log_prob = tf.log(tf.reduce_sum(self.policy_x * self.x_t, axis=1) + 1e-10)
        y_log_prob = tf.log(tf.reduce_sum(self.policy_y * self.y_t, axis=1) + 1e-10)

        a_log_prob = tf.Print(a_log_prob, [a_log_prob, tf.shape(a_log_prob)], "a_log_prob: ")


        advantage =  (self.r_t - self.value)

        advantage = tf.squeeze(advantage)
        advantage = tf.Print(advantage, [advantage, tf.shape(advantage)], "advantage: ")

        g = tf.get_default_graph()

        a_loss_policy = 1 * (- a_log_prob  * tf.stop_gradient(20*advantage) )  # maximize policy

        with g.gradient_override_map({"Identity": "CustomGrad"}):
            x_loss_policy = tf.identity((- x_log_prob * tf.stop_gradient(20*advantage) ), name="Identity")  # maximize policy
        with g.gradient_override_map({"Identity": "CustomGrad"}):
            y_loss_policy = tf.identity((- y_log_prob * tf.stop_gradient(20*advantage) ), name="Identity")  # maximize policy

        #with g.gradient_override_map({"Identity": "CustomGrad"}):
        loss_value = tf.identity(tf.square(advantage), name="Identity")    # minimize value error

        a_loss_policy = tf.Print(a_loss_policy, [a_loss_policy, tf.shape(a_loss_policy)], "a_loss_policy: ")


        a_entropy = 0.000 * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        x_entropy = 0.000 * tf.reduce_sum(self.policy_x * tf.log(self.policy_x + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        y_entropy = 0.000 * tf.reduce_sum(self.policy_y * tf.log(self.policy_y + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)


        #loss_total = tf.reduce_mean(a_loss_policy + x_loss_policy + y_loss_policy + loss_value + entropy)
        self.a_loss = tf.reduce_mean(a_loss_policy)
        self.x_loss = tf.reduce_mean(x_loss_policy)
        self.y_loss = tf.reduce_mean(y_loss_policy)
        self.v_loss = tf.reduce_mean(loss_value)

        #optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)

        #a_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)
        #x_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)
        #y_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE * 0.1)

        v_optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        #v_optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE)


        #a_minimize = a_optimizer.minimize(a_loss)
        #x_minimize = x_optimizer.minimize(x_loss)
        #y_minimize = y_optimizer.minimize(y_loss)
        #v_minimize = v_optimizer.minimize(v_loss)
        self.minimize = v_optimizer.minimize(self.a_loss + self.x_loss + self.y_loss + self.v_loss)
        #self.minimize = v_optimizer.minimize(self.v_loss)


    def train(self, a, r, s, rnn_state):

        _, a_loss, x_loss, y_loss, v_loss  = self.session.run([self.minimize, self.a_loss, self.x_loss, self.y_loss, self.v_loss],
                                                  feed_dict={self.inputs_unit_type: s[0],
                                                             self.input_player: s[1],
                                                             self.a_t : a[0],
                                                             self.x_t : a[1],
                                                             self.y_t : a[2],
                                                             self.r_t : r,
                                                             self.state_in[0]: rnn_state[0],
                                                             self.state_in[1]: rnn_state[1],
                                                             })
        return a_loss, x_loss, y_loss, v_loss

    def save(self):
        #self.saver.save(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def restore(self):
        #self.saver.restore(self.session, 'models/model-' + str(1) + '.cptk')
        pass

    def predict(self, s, batch_rnn_state):
        with self.default_graph.as_default():
            a, x, y, v, batch_rnn_state  = self.session.run([self.policy, self.policy_x, self.policy_y, self.value, self.state_out],
                                   feed_dict={self.inputs_unit_type: s[0],
                                              self.input_player: s[1],
                                              self.state_in[0] : batch_rnn_state[0],
                                              self.state_in[1]: batch_rnn_state[1]})
            return a, x, y, v, batch_rnn_state



    def get_flatten_conv(self, image_unit_type):
        # image_unit_type = tf.Print(image_unit_type, [image_unit_type], "get_flatten_conv: ")

        type_conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=image_unit_type, num_outputs=16,
                                 kernel_size=8, stride=4, padding='VALID')
        # type_conv1 = tf.Print(type_conv1, [type_conv1], "type_conv1: ")

        type_conv2 = slim.conv2d(activation_fn=tf.nn.elu,
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