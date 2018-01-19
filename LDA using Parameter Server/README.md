# LDA using JBosen Parameter Server

This is an implementation of LDA with Gibbs Sampling using a Parameter server. 

## Data

20 news-groupset (http://qwone.com/~jason/20Newsgroups/)

## Parameter Server and LDA
Parameter server (PS) is an abstraction for distributed, data-parallel ML. Each worker holds a partition of the dataset but has access to the full model parameters stored on the server machines through a fully connected bipartite network topology. Petuum's Bosen PS is used in this implementation using JBosen, a minimal Java version of Bosen. The source code and details about JBosen can be found at: https://github.com/petuum/jbosen.
Latent Dirichlet Allocation is a generative model that represents set of documents as mixtures of topics or clusters. Gibbs Sampling is used to estimate the model from the given data.
 
