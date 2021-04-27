3 Statistical Decoding (Germann et al. , 2001) presents a greedy approach to search for the translation that is most likely according to previously learned statitistical models. 
The system uses models GIZA++ and ISI ReWrite decoder (Germann et al. , 2001). 
Many existing systems for statistical machine translation (Garca-Varea and Casacuberta 2001; Germann et al. 2001; Nieen et al. 1998; Och, Tillmann, and Ney 1999) implement models presented by Brown, Della Pietra, Della Pietra, and Mercer (1993): The correspondence between the words in the source and the target strings is described by alignments that assign target word positions to each source word position. 
(2001) and Germann et al. 
The decoding algorithm that we use is a greedy one  see (Germann et al. , 2001) for details. 
