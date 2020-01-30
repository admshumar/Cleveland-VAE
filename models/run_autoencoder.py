from models import ae_class

booleans = [True, False]

for synthetic_boolean in booleans:
    for
    autoencoder = ae_class.Autoencoder(is_synthetic=synthetic_boolean).show_latent_representation()
    del autoencoder
