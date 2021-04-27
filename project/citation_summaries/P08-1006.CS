1 Introduction Full parsing analyzes the phrase structure of a sentence and provides useful input for many kinds of high-level natural language processing such as summarization (Knight and Marcu, 2000), pronoun resolution (Yang et al., 2006), and information extraction (Miyao et al., 2008).


We followed the PPI extraction method described in (Miyao et al., 2008), where two types of features were used for a SVM classifier.


We used the C&C parser (Clark and Curran, 2007), ENJU (Miyao and Tsujii, 2008), and a variant of ENJU (Hara et al., 2007) adapted for the biomedical domain (i.e., ENJU-Genia); There were a number of practical issues to consider when using parsers for this task.


3.1 Feature Vector We propose a feature vector with three types of features, corresponding to the three different kernels, which were each combined with the two parsers: the Enju 2.3.0, and KSDEP beta 1 (Miyao et al., 2008); this feature vector is used because the kernels with these parsers were shown to be effective for PPI extraction by Miwa et al.


For further improvement, we need to analyze the cause 128 positive all P R F AUC SVM-CW 1,000 5,834 60.0 71.9 65.2 89.3 L2-SVM 1,000 5,834 62.7 66.6 64.2 89.1 (Miwa et al., 2008) 1,005 5,648 60.4 69.3 64.2 (61.5) 87.9 (87.1) (Miyao et al., 2008) 1,059 5,648 54.9 65.5 59.5 (Airola et al., 2008) 1,000 5,834 52.9 61.8 56.4 84.8 (Stre et al., 2007) 1,068 5,631 64.3 44.1 52.0 (Erkan et al., 2007) 951 4,020 59.6 60.7 60.0 (Bunescu and Mooney, 2005) 65.0 46.4 54.2 Table 5: Comparison with previous PPI extraction results on the AIMed corpus.


