Thus, Nakagawa (2007) and Hall (2007) both try to overcome the limited feature scope of graph-based models by adding global features, in the former case using Gibbs sampling to deal with the intractable inference problem, in the latter case using a re-ranking scheme. 
Recent work shows that k-best maximum spanning tree (MST) parsing and reranking is also viable (Hall, 2007). 
2.1.4 Model Features Our MST models are based on the features described in (Hall, 2007); specifically, we use features based on a dependency nodes form, lemma, coarse and fine part-of-speech tag, and morphologicalstring attributes. 
3 Results and Analysis Hall (2007) shows that the oracle parsing accuracy of a k-best edge-factored MST parser is considerably higher than the one-best score of the same parser, even when k is small. 
The tree-based reranker includes the features described in (Hall, 2007) as well as features based on non-projective edge attributes explored in (Havelka, 2007a; Havelka, 2007b). 
