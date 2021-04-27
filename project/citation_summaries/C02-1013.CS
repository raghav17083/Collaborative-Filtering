We read dependency tuples directly off the output of RASP (Briscoe and Carroll, 2002b) in grammatical relation mode.1 RASP has the advantage that recall is high, although precision is potentially lower 1We used the first parse in the experiments reported here. 
Another output possibility is weighted GRs (Carroll and Briscoe, 2002); this is the unique set of GRs from the n-best GRs, each GR weighted according to the sum of the probabilities of the parses in which it occurs. 
1 Introduction RASP is a robust statistical analysis system for English developed by Briscoe and Carroll (2002). 
Therefore, high precision GR sets can be determined by thresholding on the GR weight (Carroll and Briscoe, 2002). 
Our method has been implemented using a decision-tree classifier which tests for the presence of grammatical relations (GRs) in the output of the RASP (Robust Accurate Statistical Parsing) system (Briscoe and Carroll, 2002). 
