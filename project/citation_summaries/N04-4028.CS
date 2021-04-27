This is analogous to the task of estimating record confidence using field confidence scores in information extraction (Culotta and McCallum, 2004). 
Segment confidence is estimated using constrained forward-backward (Culotta and McCallum, 2004). 
Named entities are represented using an BIO-encoding, and methods analogous to other confidence-based taggers (Culotta and McCallum, 2004; Carpenter, 2007) are used to estimate the conditional probability of tag sequences corresponding to named entities. 
(Culotta and McCallum, 2004) suggest several ways to generate confidence estimation in this framework. 
It is calculated by constrained forwardbackward algorithm (Culotta and McCallum, 2004), and confident segments are added to the dictionary in order to improve segmentation accuracy. 
