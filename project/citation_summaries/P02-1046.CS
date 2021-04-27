(2002) and Abney (2002), in which newly labelled training examples are selected using a greedy algorithm which explicitly maximises the POS taggers agreement on unlabelled data. 
for this specific task in (Abney, 2002). 
In recent work, (Abney, 2002) shows that the independence assumption can be relaxed, and co-training is still effective under a weaker independence assumption. 
Our agreement-based co-training results support the theoretical arguments of Abney (2002) and Dasgupta et al. 
3.2 Unsupervised Learning We have implemented the Greedy Agreement Algorithm (Abney, 2002) which, based on two independent views of the data, is able to learn two binary classifiers from a set of hand-typed seed rules. 
