The L1 or L2 norm is commonly used in statistical natural language processing (Gao et al., 2007).


In other words, learning with L1 regularization naturally has an intrinsic effect of feature selection, which results in an 97 efficient and interpretable inference with almost the same performance as L2 regularization (Gao et al., 2007).


lscript1-regularized log-linear models (lscript1-LLMs), on the other hand, provide sparse solutions, in which weights of irrelevant features are exactly zero, by assumingaLaplacianpriorontheweights(Tibshirani, 1996; Kazama and Tsujii, 2003; Goodman, 2004; Gao et al., 2007).


3.5 Regularization We apply lscript1 regularization (Ng, 2004; Gao et al., 2007) to make learning more robust to noise and control the effective dimensionality of the feature spacebysubtractingaweightedsumofabsolutevalues of parameter weights from the log-likelihood of the training data w = argmaxw LL(w) summationdisplay i Ci|wi| (6) We optimize the objective using a variant of the orthant-wise limited-memory quasi-Newton algorithm proposed by Andrew & Gao (2007).3 All values Ci are set to 1 in most of the experiments below, although we apply stronger regularization (Ci = 3) to reordering features.


There is usually not a considerable difference between the two methods in terms of the accuracy of the resulting model (Gao et al., 2007), but L1 regularization has a significant advantage in practice.


