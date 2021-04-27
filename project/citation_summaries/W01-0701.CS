Florian and Ngai (2001) used the same multitask learning strategy with a transformation-based learner to show that usually disjointly handled tasks perform slightly better under a joint model; the experiments there were run on POS tagging and text chunking, Chinese word segmentation and POS tagging. 
fnTBL (Ngai and Florian, 2001), an efficient implementation of TBL created by Ngai and Florian at JHU. 
To overcome the lengthy training time associated with this approach, we used fnTBL, a fast version of TBL that preserves the performance of the algorithm (Ngai and Florian, 2001). 
For systems that perform both word segmentation and POS tagging, the following results were published: Florian and Ngai (2001) report an SF of 93.55% and a TA of 88.86% on CTB data. 
Florian and Ngai (2001) extends transformationbased learning tagger to a joint tagger/chunker by modifying the objective function such that a transformation rule is evaluated on the classification of all simultaneous subtasks. 
