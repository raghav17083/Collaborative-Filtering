(2006a) that self-training with a reranking parsing model is effective for improving parser accuracy in general, and the claim of Gildea (2001) that training on in-domain data is effective for parser adaption. 
For example, the performance of a parser trained on the Wall Street Journal (newspaper text) significantly drops when evaluated on the more varied Brown (fiction/nonfiction) corpus (Gildea, 2001). 
2.3 Domain Adaptation Track One well known characteristic of data-driven parsing systems is that they typically perform much worse on data that does not come from the training domain (Gildea, 2001). 
Table 3 presents our baseline performance, compared with the Gildea (2001) results. 
The BROWN parsing model is naturally better than the WSJ model for this task, but combining the two training corpora results in a better model (as in Gildea (2001)). 
