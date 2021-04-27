More specialized methods also exist, for example for support vector machines (Musicant et al. , 2003) and for conditional random fields (Gross et al. , 2007; Suzuki et al. , 2006).


(Suzuki et al. , 2006) 88.02 (+0.82) + unlabeled data (17M  27M words) 88.41 (+0.39) + supplied gazetters 88.90 (+0.49) + add dev.


5.5 Applying F-score Optimization Technique In addition, we can simply apply the F-score optimization technique for the sequence labeling tasks proposed in (Suzuki et al. , 2006) to boost the HySOL performance since the base discriminative models pD(y|x) and discriminative combination, namely Equation (3), in our hybrid model basically uses the same optimization procedure as CRFs.


(Suzuki et al. , 2006) 94.36 (+0.06) Table 8: The HySOL performance with the F-score optimization technique on Chunking (CoNLL-2000) experiments from unlabeled data appear different from each other.


We follow (Gao et al., 2006; Suzuki et al., 2006) and approximate the metrics using the sigmoid function.


