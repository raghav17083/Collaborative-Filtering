Brutus uses the CCG parser of (Clark and Curran, 2007, henceforth the C&C parser), Charniaks parser (Charniak, 2001) for additional CFG-based features, and MALT parser (Nivre et al., 2007) for dependency features, while (Punyakanok et al., 2008) use results from an ensemble of parses from Charniaks Parser and a Collins parser (Collins, 2003; Bikel, 2004). 
The English side of the training data is parsed with an implementation of Collins Model 2 (Collins, 2003) then head-out binarized. 
To implement h negationslash=  we use a PCFG transformation emulating (Collins, 2003)s first model, in which sisters are generated conditioned on the head tag and a simple distance function (Hageloh, 2007).7 The inprocess. 
This generalization is not only of theoretical importance, but also critical in the application to state-of-theart parsers such as (Collins, 2003) and (Charniak, 2000). 
In our current work, we use Michael Collins (Collins, 2003) parser for syntactic information. 
