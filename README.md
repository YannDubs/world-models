# Pytorch implementation of the "WorldModels"

Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631. 

Credits for the implementation: https://github.com/ctallec/world-models.

Personal modifications (mostly the [factorvae.py](https://github.com/YannDubs/world-models/blob/master/factorvae.py) and [controller.py](https://github.com/YannDubs/world-models/blob/master/models/controller.py)):
* Enables the use of a disentangled VAE [FactorVAE](https://arxiv.org/abs/1802.05983)
* Enables the use of an untrained LSTM
* Adds possible gating of the previous actions to smooth out actions

Example:
https://www.youtube.com/watch?v=iUQhGXoF_RY&feature=youtu.be


