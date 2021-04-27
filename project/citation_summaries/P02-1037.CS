We also tried automatically changing numbers, symbols, and abbreviations in the training text to match how they would be read (Roark, 2002), but this did not improve accuracy and so is not discussed further.


Roark (2002), when reviewing 2SCLITE (http://www.nist.gov/speech/ tools/) by NIST is the most commonly used alignment tool.


Model n-best List/Lattice Training Size WER (%) SER (%) Oracle (50-best lattice) Lattice 7.8 Charniak (2001) List 40M 11.9 Xu (2002) List 20M 12.3 Roark (2001) (with EM) List 2M 12.7 Hall (2003) Lattice 30M 13.0 Chelba (2000) Lattice 20M 13.0 Current ( a1 1a6 16a0  a1 1) List 20M 13.1 71.0 Current ( a1 1a6 16a0  a1 1) Lattice 20M 13.1 70.4 Roark (2001) (no EM) List 1M 13.4 Lattice Trigram Lattice 40M 13.7 69.0 Current ( a1 1a6 16a0  a1 1) List 1M 14.8 74.3 Current ( a1 1a6 16a0  a1 1) Lattice 1M 14.9 74.0 Current ( a1  a1 0) Lattice 1M 16.0 75.5 Treebank Trigram Lattice 1M 16.5 79.8 No language model Lattice 16.8 84.0 Table 3: Comparison of WER for parsing HUB-1 words lattices with best results of other works.


