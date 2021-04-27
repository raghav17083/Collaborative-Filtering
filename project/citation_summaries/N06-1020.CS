4.3 Using Unlabeled Data for Parsing Recent studies on parsing indicate that the use of unlabeled data by self-training can help parsing on the WSJ data, even when labeled data is relatively large (McClosky et al., 2006a; Reichart and Rappoport, 2007). 
For English, self-training contributes 0.83% absolute improvement to the PCFG-LA parser, which is comparable to the improvement obtained from using semi-supervised training with the twostage parser in (McClosky et al., 2006). 
type system F1% D Collins (2000) 89.7 Henderson (2004) 90.1 Charniak and Johnson (2005) 91.0 updated (Johnson, 2006) 91.4 this work 91.7 G Bod (2003) 90.7Petrov and Klein (2007) 90.1 S McClosky et al. 
Theyalsoappliedself-training to domain adaptation of a constituency parser (McClosky et al., 2006b). 
Third, we hope that the improved parses of bitext will serve as higher quality training data for improving monolingual parsing using a process similar to self-training (McClosky et al., 2006). 
