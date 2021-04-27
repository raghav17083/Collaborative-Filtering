Previous research in this area includes several models which incorporate hidden variables (Matsuzaki et al., 2005; Koo and Collins, 2005; Petrov et al., 2006; Titov and Henderson, 2007). 
5 Related Work There has not been much previous work on graphical models for full parsing, although recently several latent variable models for parsing have been proposed (Koo and Collins, 2005; Matsuzaki et al. , 2005; Riezler et al. , 2002). 
A reranking parser (see also (Koo and Collins, 2005)) is a layered model: the base layer is a generative statistical PCFG parser that creates a ranked list of k parses (say, 50), and the second layer is a reranker that reorders these parses using more detailed features. 
Discriminative training with hidden variables has been handled in this probabilistic framework (Quattoni et al. , 2004; Koo and Collins, 2005), but we choose Equation 3 for efficiency. 
In (Koo and Collins, 2005), an undirected graphical model is used for parse reranking. 
