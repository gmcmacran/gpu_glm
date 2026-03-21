# What is this library?
An implementation of the [generlized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) 
that runs on an Nvidia GPU and has a the sci-kit learn interface.

The models implemented are:
* Gaussian model with identity, log, and inverse links.
* Binomial model with logit and probit links.
* Poisson model with natural log, identity, and square root links.
* Gamma model with inverse, identity, and log links.
* Inverse Gaussian model with mu^-2, inverse, identity, and log links.

# What does this library depend on?
* `numpy`: for matrices
* `scipy`: for a few mathematical operations
* `pytest`: for testing