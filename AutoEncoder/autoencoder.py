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
    Autoencoder without tied weights.
    """

    def __init__(self,
                 network_architecture,
                 transfer_fct=tf.nn.tanh,
                 # learning_rate=0.001,
                 batch_size=100,
                 weight_regularization=0.0,
                 tied=False):

        try:
            if self.sess._opened:
                print "re-initializing encoder"
                self.sess.close()
        except AttributeError:
            print "intializing encoder"

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.batch_size = batch_size
        self.tied = tied
        self.weight_regularization = weight_regularization
        self.learning_cost = []

        # tf Graph input
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.keep_prob = tf.placeholder(tf.float32)
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

        self.x_encoded = self._encoder_network(self.network_weights["weights_encoder"],
                                               self.network_weights["biases_encoder"])

        if self.tied:
            self.x_decoded = self._decoder_network(self.network_weights["weights_encoder"],
                                                   self.network_weights["biases_encoder"])

        elif 'decoder' in self.network_weights.keys():
            self.x_decoded = self._decoder_network(self.network_weights["weights_decoder"],
                                                   self.network_weights["biases_decoder"])
        else:
            raise NotImplementedError, 'tied autoencoder not implemented. need to provide decoder network layers.'

    @staticmethod
    def _initialize_weights(architecture_dict, tied):

        all_weights = dict()

        n_input = architecture_dict.get('n_input')
        if architecture_dict.has_key('encoder'):

            lst_of_encoder_layer_sizes = list(architecture_dict.get('encoder'))
            all_weights['weights_encoder'] = dict()
            all_weights['biases_encoder'] = dict()

            for idx in range(len(lst_of_encoder_layer_sizes)):
                if idx == 0:
                    print "encoder layer {}, dimensionality {} -> {}".format(idx +1, n_input, lst_of_encoder_layer_sizes[idx])
                    all_weights['weights_encoder'].update({'h{}'.format(idx+1): tf.Variable(_xavier_init(n_input, lst_of_encoder_layer_sizes[idx]))})
                else:
                    print "encoder layer {}, dimensionality {} -> {}".format(idx + 1, lst_of_encoder_layer_sizes[idx-1], lst_of_encoder_layer_sizes[idx])
                    all_weights['weights_encoder'].update(
                        {'h{}'.format(idx + 1): tf.Variable(_xavier_init(lst_of_encoder_layer_sizes[idx - 1], lst_of_encoder_layer_sizes[idx]))})

                all_weights['biases_encoder'].update({'b{}'.format(idx+1): tf.Variable(tf.zeros([lst_of_encoder_layer_sizes[idx]], dtype=tf.float32))})

        else:
            raise AttributeError, "need list of encoder layer sizes"

        if ('decoder' in architecture_dict.keys()) & (not tied):

            lst_of_decoder_layer_sizes = list(architecture_dict.get('decoder'))
            all_weights['weights_decoder'] = dict()
            all_weights['biases_decoder'] = dict()

            for idx in range(len(lst_of_decoder_layer_sizes)):
                if idx == 0:
                    print 'decoder layer {}, dimensionality {} -> {} '.format(idx+1, lst_of_encoder_layer_sizes[-1], lst_of_decoder_layer_sizes[idx])
                    all_weights['weights_decoder'].update({'h{}'.format(idx + 1): tf.Variable(_xavier_init(lst_of_encoder_layer_sizes[-1], lst_of_decoder_layer_sizes[idx]))})
                    all_weights['biases_decoder'].update({'b{}'.format(idx + 1): tf.Variable(tf.zeros([lst_of_decoder_layer_sizes[idx]], dtype=tf.float32))})
                else:
                    print 'decoder layer {}, dimensionality {} -> {} '.format(idx + 1, lst_of_decoder_layer_sizes[idx - 1], lst_of_decoder_layer_sizes[idx])
                    all_weights['weights_decoder'].update({'h{}'.format(idx + 1): tf.Variable(_xavier_init(lst_of_decoder_layer_sizes[idx - 1], lst_of_decoder_layer_sizes[idx]))})
                    all_weights['biases_decoder'].update({'b{}'.format(idx + 1): tf.Variable(tf.zeros([lst_of_decoder_layer_sizes[idx]], dtype=tf.float32))})

            if len(lst_of_decoder_layer_sizes) > 0:
                print 'decoder layer {}, dimensionality {} -> {} '.format(len(lst_of_decoder_layer_sizes) + 1, lst_of_decoder_layer_sizes[-1], n_input)
                all_weights['weights_decoder'].update(
                    {'h{}'.format(len(lst_of_decoder_layer_sizes)+1): tf.Variable(_xavier_init(lst_of_decoder_layer_sizes[-1], n_input))})
                all_weights['biases_decoder'].update(
                    {'b{}'.format(len(lst_of_decoder_layer_sizes)+1): tf.Variable(tf.zeros([n_input], dtype=tf.float32))})
            else:
                print 'decoder layer {}, dimensionality {} -> {} '.format(len(lst_of_decoder_layer_sizes) + 1,
                                                                          lst_of_encoder_layer_sizes[-1], n_input)
                all_weights['weights_decoder'].update(
                    {'h{}'.format(len(lst_of_decoder_layer_sizes) + 1): tf.Variable(
                        _xavier_init(lst_of_encoder_layer_sizes[-1], n_input))})
                all_weights['biases_decoder'].update(
                    {'b{}'.format(len(lst_of_decoder_layer_sizes) + 1): tf.Variable(
                        tf.zeros([n_input], dtype=tf.float32))})

        elif tied & ('decoder' in architecture_dict.keys()):
            raise AttributeError, "Can't have a tied autoencoder and specify a decoder network."

        return all_weights

    def _encoder_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        layers = []
        for idx in range(len(weights)):
            if idx == 0:
                layers.append(self.transfer_fct(tf.add(tf.matmul(self.x, weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx +1)])))
                layers.append(tf.nn.dropout(layers[-1], keep_prob=self.keep_prob))
            else:
                layers.append(self.transfer_fct(tf.add(tf.matmul(layers[idx-1], weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx +1)])))

        return layers[-1]

    def _decoder_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.

        layers = []
        if self.tied:
            for idx in range(len(weights)):
                print "HERE", idx, len(weights)-1 -idx
                print weights['h{}'.format(len(weights) - 0 - idx)].get_shape()
                print biases['b{}'.format(len(weights) - 0 - idx)].get_shape()
                if idx == 0:
                    # layers.append(self.transfer_fct(
                    #     tf.add(tf.matmul(self.x_encoded, weights['h{}'.format(len(weights) - 0 - idx)], transpose_b=True),
                    #            biases['b{}'.format(len(weights) - 1 - idx)])))

                    layers.append(self.transfer_fct(
                            tf.matmul(tf.sub(self.x_encoded, biases['b{}'.format(len(weights) - 0 - idx)]), weights['h{}'.format(len(weights) - 0 - idx)], transpose_b=True)))
                            # biases['b{}'.format(len(weights) - 1 - idx)])))
                else:
                    # layers.append(self.transfer_fct(
                    #     tf.add(tf.matmul(layers[len(weights) - 1 - idx], weights['h{}'.format(len(weights) - 0 - idx)], transpose_b=True),
                    #            biases['b{}'.format(len(weights) - 1 - idx)])))

                    layers.append(self.transfer_fct(
                        tf.matmul(tf.sub(layers[len(weights) - 1 - idx], biases['b{}'.format(len(weights) - 0 - idx)]),
                                  weights['h{}'.format(len(weights) - 0 - idx)], transpose_b=True)))

        elif 'decoder' in self.network_architecture.keys():
            for idx in range(len(weights)):
                if idx == 0:
                    layers.append(self.transfer_fct(
                        tf.add(tf.matmul(self.x_encoded, weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx + 1)])))
                else:
                    layers.append(self.transfer_fct(
                        tf.add(tf.matmul(layers[idx - 1], weights['h{}'.format(idx + 1)]), biases['b{}'.format(idx + 1)])))
        else:
            raise NotImplementedError('Autoencoder either has to be tied or you need to provide a decoder network configuration.')

        return layers[-1]

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss
        reconstr_loss = tf.reduce_sum(tf.squared_difference(self.x, self.x_decoded))

        weight_reg = 0
        for val in self.network_weights['weights_encoder'].itervalues():
            weight_reg = tf.reduce_sum(val)

        if not self.tied:
            for val in self.network_weights['weights_decoder'].itervalues():
                weight_reg += tf.reduce_sum(val)

        self.cost = tf.reduce_mean(reconstr_loss) + self.weight_regularization * weight_reg / (2 * self.batch_size)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X, learning_rate = None, keep_prob = None):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        if not learning_rate:
            self.learning_rate = 0.001

        if not keep_prob:
            self.keep_prob = 1.0

        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X, self.learning_rate: learning_rate, self.keep_prob: keep_prob})
        return cost

    def encode(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_encoded, feed_dict={self.x: X, self.keep_prob: 1.0})

    def decode(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_decoded,
                             feed_dict={self.x: X})

    def _monitor_layer(self, X, layer_index, network='encoder'):

        if network == 'encoder':
            weight_matrix = self.network_weights.get('weights_encoder').get('h{}'.format(layer_index))
            biases = self.network_weights.get('biases_encoder').get('b{}'.format(layer_index))
            layer_to_run = self.transfer_fct(tf.add(tf.matmul(self.x, weight_matrix), biases))

        elif network == 'decoder':
            weight_matrix = self.network_weights.get('weights_decoder').get('h{}'.format(layer_index))
            biases = self.network_weights.get('biases_decoder').get('b{}'.format(layer_index))
            layer_to_run = self.transfer_fct(tf.add(tf.matmul(self.x_encoded, weight_matrix), biases))

        else:
            raise AttributeError, "network '{}' does not exist".format(network)

        return self.sess.run(layer_to_run, feed_dict={self.x: X, self.keep_prob: 1.0})

    @staticmethod
    def _update_learning_rate(dct, epoch):

        lr = np.PINF
        for key in dct.iterkeys():
            if int(key) <= epoch:
                lr = dct[key]
        return lr

    def train(self, X, n_epochs, learning_rate=0.0001, display_step=10, keep_prob=1.0):

        data = DataSet(X, self.batch_size)

        for epoch in range(n_epochs):

            if isinstance(learning_rate, dict):
                lr = self._update_learning_rate(learning_rate, epoch)
            else:
                lr = learning_rate

            data.reset_counter()
            costs = []

            while data.has_next():
                batch_xs = data.next_batch()
                costs.append(self.partial_fit(batch_xs, lr, keep_prob))

            self.learning_cost.append(np.mean(costs))

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print "Epoch:", '{0:04d} / {1:04d}'.format(epoch + 1, n_epochs), \
                    "cost=", "{:.9f}".format(self.learning_cost[-1])


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
        self.current_batch = 0



class VariationalAutoencoder(object):

    def __init__(self,
                 network_architecture,
                 transfer_fct=tf.nn.softsign,
                 learning_rate=0.001,
                 batch_size=100):

        if self.sess._opened:
            self.sess.close()

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()

        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = self._encoder_network(network_weights["weights_recog"],
                                                                 network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._decoder_network(network_weights["weights_gener"],
                                                     network_weights["biases_gener"])

    @staticmethod
    def _initialize_weights(n_hidden_recog_1,
                            n_hidden_recog_2,
                            n_hidden_gener_1,
                            n_hidden_gener_2,
                            n_input,
                            n_z):

        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(_xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(_xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(_xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(_xavier_init(n_hidden_recog_2, n_z))}

        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        all_weights['weights_gener'] = {
            'h1': tf.Variable(_xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(_xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(_xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(_xavier_init(n_hidden_gener_2, n_input))}

        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}

        return all_weights

    def _encoder_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

        return z_mean, z_log_sigma_sq

    def _decoder_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))

        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                                       + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                                       1)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})