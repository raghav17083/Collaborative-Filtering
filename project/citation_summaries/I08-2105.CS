To determine the supersense and isa relation we use WordNet 3.0, and a set of 7,578,112 isa relations extracted by processing the page and category network of Wikipedia1 (Nastase and Strube, 2008).


The method described here uses automatically induced selectional preferences, computed from sense-untagged data, similar to Nastase (2008).


The scores are computed following a technique similar to Nastase (2008), which is illustrated using the following example: The Kawasaki drives well, steers brilliantly both under power and in tight corners  The PMW Kawasaki is involved in the following grammatical relations in the previous sentence: (drive,subject,Kawasaki) (steer,subject,Kawasaki) 913 SampleID Lemma PMW GrRole Reading act animal artifact  person  samp190 say Sun subj org-for-members 0.00056 0.01171 0.01958  0.61422  samp190 claim Sun subj org-for-members 0.00198 0.00099 0.00893  0.50211  Table 2: Grammatical annotation file enhanced with selectional preference estimates The BNC provides the collocations (drive,subject,X) and (steer,subject,Y), to determine what kind of subject drive and steer prefer, in word-POS:frequency format: drive subject chauffeur-n:12, enginen:30, car-n:62, taxi-n:13, motorist-n:10, disk-n:15, truck-n:11, man-n:75,  steer subject power-n:6, car-n:3, sportn:2, firm-n:2, boy-n:2, government-n:2, man-n:2, people-n:2  The target whose interpretation must be determined is Kawasaki.


