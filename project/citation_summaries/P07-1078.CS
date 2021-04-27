Compared with the adaptation methods of Sagae and Tsujii (2007) and Reichart and Rappoport (2007), our approach uses the information on word pairs in auto-parsed data instead of using the whole sentences as newly labeled data for training new parsers. 
Reichart and Rappoport (2007) showed that one can self-train with only a generative parser if the seed size is small. 
The self-training protocol is the same as in (Charniak, 1997; McClosky et al., 2006; Reichart and Rappoport, 2007): we parse the entire unlabeled corpus in one iteration. 
Reichart and Rappoport applied selftraining to domain adaptation using a small set of in-domain training data (Reichart and Rappoport, 2007b). 
As in Reichart and Rappoport (2007), we see large improvements when self-training on a small seed size (10%) without using the reranker. 
