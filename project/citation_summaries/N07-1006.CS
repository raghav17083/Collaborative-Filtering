In the last few years, several approaches to metric combination have been suggested (Kulesza and Shieber, 2004; Liu and Gildea, 2007; Albrecht and Hwa, 2007a). 
Liu and Gildea (2007) recently introduced Maximum Correlation Training (MCT), a search procedure that follows the gradient of the formula for correlation coefficient (r). 
Our approach is similar to that of Liu and Gildea (2007) except that in our case all the metrics in the combination are equally important3. 
Canonical correlation finds an nlong vector x such that x = argmax ( nsummationdisplay j=1 aijxj,bi), (1) where (x,y) is the Pearson correlation between x and y. A similar approach has been used by Liu and Gildea (2007) in the application of machine translation metrics, where they use a gradient optimization method to solve the maximization problem. 
As an alternative to methods based on machine learning techniques, Liu and Gildea (2007) suggested a simpler approach based on linear combinations of metrics. 
