Whittaker and Raj (2001) use pruning, quantization and difference encoding to store language model parameters in as little as 4 bits per value, reducing language model sizes by to 60% with minimal loss in recognition performance. Federico and Bertoldi (2006) show that the performance of an SMT system does not suffer if LM parameters are quantized into 256 distinct classes (8 bits per value).


(Federico and Bertoldi, 2006) also used single machine and fewer bits to store the LM probability by using efficient prefix trees.


However, recent work by Och (2005) and Federico and Bertoldi (2006) has shown that the statistics used by phrase-based systems are not very precise.


(Federico and Bertoldi, 2006) showed that best results were achieved with the so-called binning method.


Since the use of clusters of machines is not always practical (or affordable) for SMT applications, an alternative strategy is to find more efficient ways to store the LM in the working memory of a single machine, for instance by using efficient prefix trees and fewer bits to store the LM probability (Federico and Bertoldi, 2006).


