For example, (Duh and Kirchhoff, 2004) showed that factored language models, which consider morphological features and use an optimized backoff policy, yield lower perplexity.


The FLM baseline is a handoptimized 3-gram FLM (Model 5); we also tested an FLM optimized with a genetic algorithm as de3 # Model ECA dev ECA eval Turkish dev Turkish eval no unk w/unk no unk w/unk no unk w/unk no unk w/unk 1 Baseline 3-gram 191 176 183 172 827 569 855 586 2 Class-based LM 221 278 219 269 1642 1894 1684 1930 3 1) & 2) 183 169 178 167 790 540 814 555 4 Word-based NLM 208 341 204 195 1510 1043 1569 1067 5 1) & 4) 178 165 173 162 758 542 782 557 6 Word-based NLM 202 194 204 192 1991 1369 2064 1386 7 1) & 6) 175 162 173 160 754 563 772 580 8 hand-optimized FLM 187 171 178 166 827 595 854 614 9 1) & 8) 182 167 174 163 805 563 832 581 10 genetic FLM 190 188 181 188 761 1181 776 1179 11 1) & 10) 183 166 175 164 706 488 720 498 12 factored NLM 189 173 190 175 1216 808 1249 832 13 1) & 12) 169 155 168 155 724 487 744 500 14 1) & 10) & 12) 165 155 165 154 652 452 664 461 Table 2: Perplexities for baseline backoff LMs, FLMs, NLMs, and LM interpolation scribed in (Duh and Kirchhoff, 2004) (Model 6).


Since the space of different combinations is too large to be searched exhaustively, we use a guided search procedure based on Genetic Algorithms (Duh and Kirchhoff, 2004), which optimizes the FLM structure with respect to the desired criterion.


