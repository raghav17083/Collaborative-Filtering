The algorithm is outlined as follows: after clustering N references into m clusters, for each cluster C k  in {C 1 , C 2 , , C m }, we calculate the score of each feature for C k  and choose features as the label of C k  whose scores rank top N. In particular, the score caculated in this paper is different from Pedersen and Kulkarnis (2006). 
The kmeans algorithm is used for clustering the contexts, where the number of clusters is automatically discovered using the Adapted Gap Statistic (Pedersen and Kulkarni, 2006). 
Pedersen and Kulkarni (2006) selected the top N non-stopping word features from texts grouped in a cluster as label. 
We leave it as our future work to test methods of automatically determining the cluster number (Pedersen and Kulkarni, 2006; Blei and Jordan, 2006). 
In the study of cluster stopping measures by Pedersen and Kulkarni (2006), the criterion functions defines text similarity based on cosine value of the angle between vectors. 
