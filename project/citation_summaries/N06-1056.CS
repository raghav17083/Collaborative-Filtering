3 The Semantic Parsing Algorithm Our work is based on the WASP semantic parsing algorithm(WongandMooney,2006),whichtranslates NL sentences into MRs using an SCFG. 
For some domains, this problem can be avoided by transforming a logical language into a variable-free, functional language (e.g. the GEOQUERY functional querylanguageinWongandMooney(2006)). 
We use a maximum-entropy model similar to that of Zettlemoyer and Collins (2005) and Wong and Mooney (2006). 
3 Meaning Representation We restrict our meaning representation (MR) formalism to a variable free version as presented in (Wong and Mooney, 2006; Kate et al., 2005). 
In our previous work (Wong and Mooney, 2006), semantic parsing is cast as a machine translation task, where an SCFG is used to model the translation of an NL into a formal meaning-representation language (MRL). 
