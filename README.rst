DSTK: The Data Science Toolkit
==============================

This repository contains advanced tools for Data Scientists to use them as off-the-shelf solution.
Feel free to use and contribute.

Setup
-----

To setup, clone the repository and run

.. code:: python

    python setup.py install

Content
-------

- Autoencoder:
    The AutoEncoder package contains a simple implementation using TensorFlow under the hood.
    It provides support for arbitrarily deep networks and tied as well as untied versions of the encoder.
    To create an AutoEncoder you need to provide a network architecture dictionary

    .. code:: python

        network_architecture ={
            'n_input': 100,
            'n_compressed': 2,
            'encoder': [100, 10],
            'decoder': [10, 100]
        }

    where n_compressed is the dimension of the compressed representation. The encoder is set up by calling

    .. code:: python

        sae = SimpleAutoencoder(network_architecture,
                                weight_regularization=1.0,
                                bias_regularization=0.1,
                                batch_size=25,
                                transfer_fct=tf.nn.softsign,
                                tied=False)

    To train the encoder you can provide a learning rate schedule

    .. code:: python

        schedule = {
            0: 0.001,
            250: 0.00025,
            2500: 0.0001,
            4000: 0.00005,
        }

    The training is then started with

    .. code:: python

        sae.train(data, 5000, learning_rate=schedule, display_step=100)

    The training is incremental, so you can continue training for more epochs after it finished, by just calling `train()` again. It will continue with the state it is in.
    To encode/decode a new data point you simply can call

    .. code:: python

        sae.encode(data)
        sae.decode(compressed_data)

    To monitor the progress during training, there is an internal recording dictionary that contains the losses of epochs, learning rate schedule and total number of training epochs

    .. code:: python

        sae._recording

    The implementation of TensorFlow also allows to produce the output of any arbitrary layer in the encoder. This can be done by calling

    .. code:: python

        Y = sae._monitor_layer(data, layer_num, network='encoder')

    where `layer_num` is the layer index of the `encoder` or `decoder` network.