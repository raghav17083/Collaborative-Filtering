In fact, rather than joint parsing per se, Burkett and Klein (2008) resort to separate monolingual parsing and bilingual reranking over k2 tree pairs, which covers a tiny fraction of the whole space (Huang, 2008). 
In Table 9, Petrov and Klein (2007) trained their model on CTB articles 1-270 and 400-1151, and Burkett and Klein (2008) used the same CTB articles and parse trees of their English translation (from the English Chinese Translation Treebank) as training data. 
System  F1-Measure I Charniak (2000) 80.85 Petrov and Klein (2007) 83.13 B Burkett and Klein (2008) 1  84.24 C Our method 85.45  Table 11. 
Table 5 shows the split of this data into training, development, and test subsets according to Burkett and Klein (2008). 
By contrast, Burkett and Klein (2008) reported their approach of monolingual k-best parsing followed by bilingual k2-best reranking to be 3.8 times slower than monolingual parsing. 
