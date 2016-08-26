import numpy as np
import tensorflow as tf


def _xavier_init(fan_in, fan_out, constant=1):
    """
    Xavier initialization of network weights.
    For some explanations see:
        - http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        - https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class SimpleAutoencoder(object):
    """
    Autoencoder implementation based on inputs from https://jmetzen.github.io/2015-11-27/vae.html.
    See the project README on how to use it.
    """

    def __init__(self,
                 network_architecture,
                 transfer_fct=tf.nn.tanh,
                 batch_size=100,
                 weight_regularization=0.0,
                 bias_regularization=0.0,
                 tied=False):

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.batch_size = batch_size
        self.weight_regularization = weight_regularization
        self.bias_regularization = bias_regularization
        self.tied = tied
        self._encoder_layers = []
        self._decoder_layers = []
        self._recording = {'epoch': 0,
                           'learning_rate_schedule': dict(),
                           'batch_cost': []}

        # tf Graph input
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self._create_network()

        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        num_cores = 6
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'CPU': num_cores},
                                                                inter_op_parallelism_threads=num_cores,
                                                                intra_op_parallelism_threads=num_cores))
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencoder network weights and biases
        self.network_weights = self._initialize_weights(self.network_architecture, self.tied)

        if self.tied:
            self.x_encoded, self.x_decoded = self._tied_encoder_network(self.network_weights["weights_encoder"],
                                                                        self.network_weights["biases_encoder"],
                                                                        self.network_weights["biases_decoder"])

        else:

            assert 'decoder' in self.network_architecture.keys(), "Need to provide decoder network layer configuration for untied encoder."

            self.x_encoded = self._encoder_network(self.network_weights["weights_encoder"],
                                                   self.network_weights["biases_encoder"])

            self.x_decoded = self._decoder_network(self.network_weights["weights_decoder"],
                                                   self.network_weights["biases_decoder"])


    @staticmethod
    def _create_matrix_and_bias_sizes(n_input, lst_of_encoder_layer_sizes, n_compressed):
        matrix_dims = zip([n_input] + lst_of_encoder_layer_sizes, lst_of_encoder_layer_sizes + [n_compressed])
        bias_dims = [tup[1] for tup in matrix_dims]
        return matrix_dims, bias_dims

    @staticmethod
    def _create_layers(component, lst_of_component_layer_sizes, n_input, n_compressed):

        all_weights = dict()
        if len(lst_of_component_layer_sizes) > 0:

            matrix_dims, bias_dims = SimpleAutoencoder._create_matrix_and_bias_sizes(n_input,
                                                                                     lst_of_component_layer_sizes,
                                                                                     n_compressed)
            all_weights['weights_{}'.format(component)] = dict()
            all_weights['biases_{}'.format(component)] = dict()

            for idx, dim in enumerate(matrix_dims):
                print "{} layer {}, dimensionality {} -> {}".format(component, idx + 1, dim[0], dim[1])
                all_weights['weights_{}'.format(component)].update(
                    {'h{}'.format(idx + 1): tf.Variable(_xavier_init(dim[0], dim[1]))})

                all_weights['biases_{}'.format(component)].update(
                    {'b{}'.format(idx + 1): tf.Variable(tf.zeros([bias_dims[idx]], dtype=tf.float32))})

            return all_weights

        else:
            raise AttributeError("need list of {} layer sizes".format(component))

    @staticmethod
    def _initialize_weights(architecture_dict, tied):

        all_weights = dict()

        n_input = architecture_dict.get('n_input')
        n_compressed = architecture_dict.get('n_compressed')
        all_weights.update(SimpleAutoencoder._create_layers('encoder', architecture_dict.get('encoder'), n_input, n_compressed))

        if tied:
            lst_of_dims = architecture_dict.get('encoder')

            # List `reverse` manipulates IN-PLACE and returns None. Hence we cannot put this inline :(
            lst_of_dims.reverse()
            all_weights.update(SimpleAutoencoder._create_layers('decoder', lst_of_dims, n_compressed, n_input))

        else:
            assert 'decoder' in architecture_dict.keys(), "Need to provide decoder network layer configuration for untied encoder."

            all_weights.update(SimpleAutoencoder._create_layers('decoder', architecture_dict.get('decoder'), n_compressed, n_input))

        return all_weights

    def _tied_encoder_network(self, weights, biases_encoder, biases_decoder):

        layers = []
        for idx in range(len(weights)):
            if idx == 0:
                layers.append(self.transfer_fct(
                    tf.add(tf.matmul(self.x, weights['h{}'.format(idx + 1)]), biases_encoder['b{}'.format(idx + 1)])))
            else:
                layers.append(self.transfer_fct(
                    tf.add(tf.matmul(layers[idx - 1], weights['h{}'.format(idx + 1)]), biases_encoder['b{}'.format(idx + 1)])))

        for idx in range(len(weights)):
            if idx == 0:
                layers.append(self.transfer_fct(
                    tf.add(tf.matmul(layers[-1], weights['h{}'.format(len(weights) - idx)], transpose_b=True), biases_decoder['b{}'.format(idx + 1)])))
            else:
                layers.append(self.transfer_fct(
                    tf.add(tf.matmul(layers[len(weights) + idx -1], weights['h{}'.format(len(weights) - idx)], transpose_b=True),
                           biases_decoder['b{}'.format(idx + 1)])))

        self._encoder_layers = layers[:len(weights)]
        self._decoder_layers = layers[len(weights):]
        return layers[len(weights) - 1], layers[-1]

    def _encoder_network(self, weights, biases):

        layers = []
        for idx in range(len(weights)):
            if idx == 0:
                layers.append(self.transfer_fct(tf.add(tf.matmul(self.x, weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx +1)])))
            else:
                layers.append(self.transfer_fct(tf.add(tf.matmul(layers[idx-1], weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx +1)])))

        self._encoder_layers = layers
        return layers[-1]

    def _decoder_network(self, weights, biases):

        layers = []
        for idx in range(len(weights)):
            if idx == 0:
                layers.append(self.transfer_fct(
                    tf.add(tf.matmul(self.x_encoded, weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx + 1)])))
            else:
                layers.append(self.transfer_fct(
                    tf.add(tf.matmul(layers[idx - 1], weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx + 1)])))

        self._decoder_layers = layers
        return layers[-1]

    def _create_loss_optimizer(self):

        # The reconstruction loss
        reconstr_loss = tf.reduce_sum(tf.squared_difference(self.x, self.x_decoded))

        # Weight matrix regularization loss
        weight_reg = 0
        for val in self.network_weights['weights_encoder'].itervalues():
            weight_reg += tf.reduce_sum(tf.square(val))

        for val in self.network_weights['weights_decoder'].itervalues():
            weight_reg += tf.reduce_sum(tf.square(val))

        # Bias vector regularization loss
        bias_reg = 0
        for val in self.network_weights['biases_encoder'].itervalues():
            bias_reg += tf.reduce_sum(tf.square(val))

        for val in self.network_weights['biases_decoder'].itervalues():
            bias_reg += tf.reduce_sum(tf.square(val))

        self.cost = tf.reduce_mean(reconstr_loss) + (self.weight_regularization * weight_reg + self.bias_regularization * bias_reg) / (2 * self.batch_size)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X, learning_rate=None):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        if not learning_rate:
            self.learning_rate = 0.001

        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X, self.learning_rate: learning_rate})
        return cost

    def encode(self, X):
        """Transform data by mapping it into the latent space."""
        return self.sess.run(self.x_encoded, feed_dict={self.x: X})

    def decode(self, X):
        return self.sess.run(self.x_decoded, feed_dict={self.x: X})

    def _monitor_layer(self, X, layer_index, network='encoder'):

        if network == 'encoder':
            layer_to_run = self._encoder_layers[layer_index]

        elif network == 'decoder':
            layer_to_run = self._decoder_layers[layer_index]

        else:
            raise AttributeError("network '{}' does not exist".format(network))

        return self.sess.run(layer_to_run, feed_dict={self.x: X})

    def _update_learning_rate(self, dct, epoch):

        epoch_key = max(k for k in dct if k <= epoch)
        if self._recording['epoch'] <= 1:
            self._current_lr = dct[epoch_key]
            self._recording['learning_rate_schedule'].update({self._recording['epoch'] - 1: self._current_lr})

        if dct[epoch_key] != self._current_lr:
            self._current_lr = dct[epoch_key]
            self._recording['learning_rate_schedule'].update({self._recording['epoch'] - 1: self._current_lr})

        return self._current_lr

    def train(self, X, n_epochs, learning_rate=0.0001, display_step=10):

        data = DataSet(X, self.batch_size)

        for epoch in range(n_epochs):

            self._recording['epoch'] += 1

            if isinstance(learning_rate, dict):
                lr = self._update_learning_rate(learning_rate, epoch)
            else:
                lr = learning_rate

            data.reset_counter()
            costs = []

            while data.has_next():
                batch_xs = data.next_batch()
                costs.append(self.partial_fit(batch_xs, lr))

            self._recording['batch_cost'].append(np.mean(costs))

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print "Epoch:", '{0:04d} / {1:04d}'.format(epoch + 1, n_epochs), \
                    "cost=", "{:.9f}".format(self._recording['batch_cost'][-1])


class DataSet(object):

    def __init__(self, data, batch_size = 100):
        self.data = data
        self.n_samples = self.data.shape[0]
        self.n_dimensions = self.data.shape[1]
        self.batch_size = batch_size
        self.total_batches = int(self.n_samples / self.batch_size)
        self.current_batch = 0

    def next_batch(self):
        batch_lower_idx = self.current_batch * self.batch_size
        batch_upper_idx = (self.current_batch + 1) * self.batch_size

        self.current_batch += 1

        return self.data[batch_lower_idx:batch_upper_idx, :]

    def has_next(self):
        return self.current_batch < self.total_batches

    def reset_counter(self):
        # Shuffle data set on re-initialization
        # Note that shuffle does this IN-PLACE. Sad :(
        np.random.shuffle(self.data)
        self.current_batch = 0
