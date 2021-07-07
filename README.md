# Neural Process Family

Pytorch implementation of neural process families (CNP, NP, ANP, Meta-Fun) on both regression and classification.

Neural Process Family is devised to alternate the gaussian process with a scalable neural network.
Specifically, given a set of input-output pairs {Cx, Cy} and some input Tx, the model is expected to credibly estimate its corresponding output Ty.
Hence, it follows the problem setting of few-shot learning and exploits an encoder-decoder pipeline.
A permutation invariant set encoding r is first extracted from the {Cx, Cy}, and Tx is then feedforwarded to decoder along with r to estimate the parameters of the distribution of Ty.

Conditional Neural Process(CNP) was the first instantiation of the studies, which is trained to maximize the marginal likelihood.
Neural Process(NP) is a simple extension to CNP by incorporating the stochastic latent variable following Variational AutoEncoder(VAE).
Since the marginal likelihood is no more tractable, variational inference technique is applied and the model is trained by maximizing the Evidence Lower Bound(ELBO).
Based on the Kolmogorov Extension Theorem, NP is proven to be a stochastic process, however, many complex modules are required to avoid underfitting due to difficulty on approximate inference in nature of probabilistic models.

Attentive Neural Process(ANP) is one of the follow-up studies based on this idea such that multi-head attention and self-attention introduced in Transformer is used to consider the dependency between the set elements.
Meta-Fun further bridges to the functional gradient descent that implictly relaxes to the infinite dimensional representation space.
