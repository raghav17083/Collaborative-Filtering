3 The NatLog system Our implementation of natural logic, the NatLog system, uses a multi-stage architecture like those of (Marsi and Krahmer, 2005; MacCartney et al., 2006), comprising (1) linguistic analysis, (2) alignment, (3) lexical entailment classification, (4) entailment projection, and (5) entailment composition. 
4 System overview Our system is based on the stage architecture of the Stanford RTE system (MacCartney et al., 2006), but adds a stage for event coreference decision. 
Some authors have already designed similar matching techniques, such as the ones described in (MacCartney et al. , 2006) and (Snow et al. , 2006). 
We can also observe a trend in recent work in textual entailment that more emphasis is put on explicit learning of the syntactic graph mapping between the entailed and entailed-by sentences (MacCartney et al. , 2006). 
5.1 The Stanford RTE System We base our experiments on the Stanford RTE system which uses a staged architecture (MacCartney et al., 2006). 
