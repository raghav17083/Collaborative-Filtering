3.2 Semantic Kernels It has been shown that semantic information is fundamental for improving the accuracy and reducing the amount of training data in many natural language tasks, including fine-grained classification of named entities (Fleischman and Hovy, 2002), question classification (Li and Roth, 2005), text categorization (Giozzo and Strapparava, 2005), word sense disambiguation (Gliozzo et al., 2005).


Results reported by R(90) Top-10 (Gao et al. , 2003) 88.42 93.07 (Kim et al. , 2005) 87.11 92.21 (Gliozzo and Strapparava, 2005) 92.80 Table 2: F1 results reported for the Reuters-21578 ModApte split.


In practice, each text segment is represented in the LSA space by summing up the normalized LSA vectors of all the constituent words, using also a tf.idf weighting scheme (Gliozzo and Strapparava, 2005).


