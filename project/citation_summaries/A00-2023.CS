Like Nitrogen, the HALogen system (Langkilde, 2000; Langkilde-Geary, 2002a, 2002b) uses word n-grams, but it extracts the best-scoring surface realizations efficiently from a packed forest by constraining the search first within the scope of each constituent.


The PTB to DSIF transformation pipeline includes the following stages, inspired by Langkilde-Gearys (2002b) description: A. Deserialize the tree B. Label heads, according to Charniaks head labeling rules (Charniak, 2000) C. Remove empty nodes and flatten any remaining empty non-terminals D. Relabel heads to conform more closely to the head conventions of NLPWin E. Label with logical roles, inferred from PTB functional roles F. Flatten to maximal projections of heads (MPH), except in the case of conjunctions G. Flatten non-branching non-terminals H. Perform dictionary look-up and morphological analysis I. Introduce structure for material between paired delimiters and for any coordination not already represented in the PTB J. Remove punctuation K. Remove function words L. Map the head of each maximal projection to a dependency node, and map the heads modifiers to the first nodes dependents, thereby forming a complete dependency tree.


Prominent examples of surface realizers in the generate-and-select paradigm include Nitrogen/Halogen (Langkilde, 2000; Langkilde-Geary, 2002) and Fergus (Bangalore and Rambow, 2000).


[Langkilde, 2000] Irene Langkilde.


Insofar as it is a broad coverage generator, which has been trained and tested on sections of the WSJ corpus, our generator is closer to the generators of (Bangalore and Rambow, 2000; Langkilde-Geary, 2002; Nakanishi et al. , 2005) than to those designed for more restricted domains such as weather forecast (Belz, 2007) and air travel domains (Ratnaparkhi, 2000).


