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

    The training is incremental, so you can continue training for more epochs after it finished, by just calling :code:`train()` again. It will continue with the state it is in.
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

    where :code:`layer_num` is the layer index of the :code:`encoder` or :code:`decoder` network.


- Generalized Additive Model (GAM) with Gradient Boosting:
    This package provides an implementation of a GAM algorithm proposed in the paper `Intelligible models for classification and regression`_
    by Yin Lou, Rich Caruana and Johannes Gehrke (Proceedings KDD'12). This model has successfully been used in understanding clinical data for
    `Predicting Pneumonia Risk and Hospital 30-day Readmission`_

    The basic idea is to learn the univariate shape function of an attribute, while optimizing the overall generalized model

    .. math::
        g(x) = f_1(x_1) + f_2(x_2) + ... + f_n(x_n)

    where :math:`g(x)` is a nonlinear function such as the :math:`logit`. The shape function :math:`f_i` can be nonlinear themselves and are found using
    function approximation using a `greedy gradient boosting machine`_.

    .. _Intelligible models for classification and regression: https://dl.acm.org/citation.cfm?doid=2339530.2339556
    .. _greedy gradient boosting machine: https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    .. _Predicting Pneumonia Risk and Hospital 30-day Readmission: http://doi.acm.org/10.1145/2783258.2788613

    To instantiate a GAM use

    .. code:: python

        gam = GAM(max_leaf_nodes=10, min_samples_leaf=75)

    :code:`GAM` leverages a scikit-learn :code:`sklearn.tree.DecisionTreeRegressor` under the hood and hence exposes all its :code:`**kwargs`.
    To train the :code:`GAM` use

    .. code:: python

        gam.train(x_train, y_train, n_iter=10, display_step=2, leaning_rate=0.002)

    The algorithm can be trained iteratively, i.e. if it's not yet converged, calling :code:`train()` again will use its last state to continue the training.
    Moreover, there is a training recording, that stores the number of epochs and various classification metrics (using the training set)

    .. code:: python

        gam._recording


- Bolasso:
    The Bolasso package provides and implementation of the Bolasso feature selection technique, based on the article `Model consistent Lasso estimation through the bootstrap`_
    by F. R. Bach.

    This feature selection wrapper trains several sklearn LogisticRegressionCV classifiers with L1-penalty on a bootstrapped subset of the data.
    It keeps a running tap on the number of occurences a given feature appeared throughout all iterations.

    To instantiate the selector we run

    .. code:: python

        b = bl.Bolasso(bootstrap_fraction=0.5)

    We can then fit the selector using

    .. code:: python

        b.fit(data_df, target_series, epochs=5)

    If the results are not yet satisfactory, you can call this again and it continues to train the same Bolasso selector.
    To get the individual feature statistics we call

    .. code:: python

        b.get_feature_stats()

    .. _Model consistent Lasso estimation through the bootstrap: http://dl.acm.org/citation.cfm?id=1390161