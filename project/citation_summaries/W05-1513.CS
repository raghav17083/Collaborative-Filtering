(2004), Nivre and Scholz (2004), and Sagae and Lavie (2005). 
3.4 The Training Set Our training set I contains all inferences considered in every state along the correct path for each goldstandard parse tree (Sagae & Lavie, 2005).7 This method of generating training examples does not require a working parser and can be run prior to any training. 
Sagae and Lavie (2005) have shown that this algorithm has linear time complexity, assuming that classification takes constant time. 
One extrapolation is to a very fast stochastic parser by Sagae and Lavie (2005). 
3.3 The Training Set We choose a single correct path from each training parse tree, and the training examples correspond to all candidate inferences considered in every state along this path.4 In the deterministic setting there is only one correct path, so example generation is identical to that of Sagae and Lavie (2005). 
