Also the use of lossy data structures based on Bloom filters has been demonstrated to be effective for LMs (Talbot and Osborne, 2007a; Talbot and Osborne, 2007b). 
Note that unlike the constructions in (Talbot and Osborne, 2007b) and (Church et al., 2007) no errors are possible for ngrams stored in the model. 
Wehope the present work will, together with Talbot and Osborne (2007), establish the Bloom filter as a practical alternative to conventional associative data structures used in computational linguistics. 
There also have been prior work on maintaining approximate counts for higher-order language models (LMs) ((Talbot and Osborne, 2007a; Talbot and Osborne, 2007b; Talbot and Brants, 2008)) operates under the model that the goal is to store a compressed representation of a disk-resident table of counts and use this compressed representation to answer count queries approximately. 
We have also implemented a Bloom Filter LM in Joshua, following Talbot and Osborne (2007). 
