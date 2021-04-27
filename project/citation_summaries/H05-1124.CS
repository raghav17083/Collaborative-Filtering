Since the CRF model is one of the successful models in sequential labeling tasks (Lafferty et al. 2001; Sha & Pereira 2003; McDonald et al. 2005), in this section, we also compare LDCRFs with CRFs. 
We used a k-best version of the MIRA algorithm (Crammer, 2004; McDonald et al. , 2005). 
The training algorithm we propose in this paper is based on the K-best MIRA algorithm which has been used earlier in structured prediction problems (McDonald et al., 2005a; McDonald et al., 2005b). 
In particular, we use a generalized version of MIRA (Crammer et al., 2005; McDonald, 2006) that can incorporate k-best decoding in the update procedure. 
This algorithm is really an extension of Viterbi to the case when scores factor over dynamic substrings of the text (Sarawagi and Cohen, 2004; McDonald et al. , 2005a). 
