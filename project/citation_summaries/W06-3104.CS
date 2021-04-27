Ourmodelisthusa form of quasi-synchronous grammar (QG) (Smith and Eisner, 2006a). 
Following Smith and Eisner (2006), we adopt the view that the syntactic structure of sentences paraphrasing some sentence s should be inspired by the structure of s. Because dependency syntax is still only a crude approximation to semantic structure, we augment the model with a lexical semantics component, based on WordNet (Miller, 1995), that models how words are probabilistically altered in generating a paraphrase. 
3.1 Background Smith and Eisner (2006) introduced the quasisynchronous grammar formalism. 
Thus, our generative model is a quasi-synchronous grammar, exactly as in (Smith and Eisner, 2006a).3 When training on target sentences w, therefore, we tune the model parameters to maximize notsummationtextt p(t,w) as in ordinary EM, but rather 3Our task here is new; they used it for alignment. 
3 Quasi-Synchronous Grammar For a formal description of QG, we recommend Smith and Eisner (2006). 
