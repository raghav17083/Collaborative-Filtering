2 The Model 2.1 A Conditional Random Field Context Free Grammar (CRF-CFG) Our parsing model is based on a conditional random field model, however, unlike previous TreeCRF work, e.g., (Cohn and Blunsom, 2005; Jousse et al., 2006), we do not assume a particular tree structure, and instead find the most likely structure and labeling.


Also in this issue, Toutanova, Haghighi, and Manning apply re-ranking to select the best among a set of candidate complete solutions produced by a base SRL system.Finally, probabilistic models have also been applied to produce the structured output, for example, generative models (Thompson, Levy, and Manning 2003), sequence tagging with classiers (M`arquez et al. 2005; Pradhan et al.2005b), and Conditional Random Fields on tree structures (Cohn and Blunsom 2005).These approaches at a global level may demand considerable extra computation, but current optimization techniques help solve them quite efciently.


The model can be used for tasks like syntactic parsing (Finkel et al., 2008) and semantic role labeling (Cohn and Blunsom, 2005).


Regarding novel learning paradigms not applied in previous shared tasks, we find Relevant Vector Machine (RVM), which is a kernelbased linear discriminant inside the framework of Sparse Bayesian Learning (Johansson and Nugues, 2005) and Tree Conditional Random Fields (T-CRF) (Cohn and Blunsom, 2005), that extend the sequential CRF model to tree structures.


