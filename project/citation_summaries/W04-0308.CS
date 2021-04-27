The parser is a bottom-up deterministic dependency parser based on the algorithm proposed by (Nivre, 2004). 
The rst strictly incremental parser of this kind was described in Nivre (2003) and used for classier-based parsing of Swedish by Nivre, Hall, and Nilsson (2004) and English by Nivre and Scholz (2004). 
4 Related work The work presented here builds on the dependency parsing work of Nivre (2004), as discussed in section 2, on the work of Nivre and Nilsson (2006) on pseudo-projective transformations, and on the work of Sagae and Tsujii (2007) in using a beam search in shift-reduce dependency parsing using maximum entropy classifiers. 
We first describe how the basic shift-reduce bottom-up dependency parsing algorithm described by Nivre (2004) can be modified to allow multiple heads per word. 
In this respect such a model is very restrictive and suffers from the pitfalls of the incremental processing (Nivre, 2004). 
