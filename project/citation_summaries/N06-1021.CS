4.1.2 Comparative results of English data Table 2 shows the performance of the systems that were compared, where Y&M2003 refers to the parser of Yamada and Matsumoto (2003), CO2006 refers to the parser of Corston-Oliver et al. 
1 Introduction A recent development in data-driven parsing is the use of discriminative training methods (Riezler et al. , 2002; Taskar et al. , 2004; Collins and Roark, 2004; Turian and Melamed, 2006). 
2 Dependency Parsing with Passive-Aggressive Learning This section presents the modification of PassiveAggressive Learning (PA) (Crammer et al. , 2006) for dependency parsing. 
The number of categories assigned to each word is determined by a parameter  in the supertagger: all categories are assigned whose forward-backward probabilities are within  of the highest probability category (Curran et al. , 2006). 
(2006), Hall2006 refers to the parser of Hall et al. 
