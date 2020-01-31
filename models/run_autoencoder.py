from models import ae_class

booleans = [True, False]
restrictions = [[1,2],[1,3],[2,3]]
cluster_sizes = [3,4]
latent_space_dimension_exponents = [2,4]
augmentation_sizes = [10,100]
covariance_coefficients = [0.1, 1, 2]
batch_sizes = [32, 64, 128]
learning_rates = [1e-3, 1e-4]
activation_functions = ['relu', 'tanh', 'elu', 'softmax', 'sigmoid']
dropout_rates = [0.5, 0.25]

for depth in booleans:
    for clusters in cluster_sizes:
        for layer_boolean in booleans:
            for exponent in latent_space_dimension_exponents:
                for augmentation_boolean in booleans:
                    for covariance in covariance_coefficients:
                        for batch_size in batch_sizes:
                            for learning_rate in learning_rates:
                                for batch_norm_boolean in booleans:
                                    for dropout_boolean in booleans:
                                        for encoder_activation in activation_functions:
                                            for decoder_activation in activation_functions:
                                                for dropout_rate in dropout_rates:
                                                    autoencoder = ae_class.Autoencoder(deep=depth,
                                                                                       number_of_clusters=clusters,
                                                                                       enable_stochastic_gradient_descent=True,
                                                                                       has_custom_layers=layer_boolean,
                                                                                       exponent_of_latent_space_dimension=exponent,
                                                                                       covariance_coefficient=covariance,
                                                                                       batch_size=batch_size,
                                                                                       learning_rate=learning_rate,
                                                                                       enable_batch_normalization=batch_norm_boolean,
                                                                                       enable_dropout=dropout_boolean,
                                                                                       encoder_activation=encoder_activation,
                                                                                       decoder_activation=decoder_activation,
                                                                                       dropout_rate=dropout_rate
                                                                                       ).show_latent_representation()
                                                    del autoencoder
