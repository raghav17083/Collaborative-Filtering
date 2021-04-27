3 Name tagger overview We assessed the name tagger described in Mota and Grishman (2008) to recognize names of people, organizations and locations.


In particular, we showed that the performance of a name tagger based on co-training decays as the time gap between training data (seeds and unlabeled data) and test data increases (Mota and Grishman, 2008).


Co-training is one of thosemethods,andhasbeenextensivelystudiedin NLP(NigamandGhani,2000; PierceandCardie, 2001; Ng and Cardie, 2003; Mota and Grishman, 2008).


Concerning variations over longer periods of time,weobservedthattheperformance ofasemisupervised name tagger decays over a period of eight years, which seems to be directly related withthefactthatthetextsusedtotrainandtestthe tagger also show a tendency to become less similar over time (Mota and Grishman, 2008); Batista etal.(2008)alsoobserved adecaying tendency in the performance of a system for recovering capitalization over a period of six years, proposing to retrain a MaxEnt model using additional contemporarywrittentexts.


