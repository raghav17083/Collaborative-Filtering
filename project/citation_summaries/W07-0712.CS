IRSTLM (Federico and Cettolo, 2007) also performs well, but the quantized model on the 21.2 data did not improve over the smaller quantized model6.


8-bit quantization was performed for probabilities and back-off coefficients using a simple binning approach (Federico and Cettolo, 2007).


IRSTLM (Federico and Cettolo, 2007) offers the option to use a custom page manager that relegates part of the structure to disk via memory-mapped files.


We have also used TPTs to encode n-gram count databases such as the Google 1T web n-gram database (Brants and Franz, 2006), but are not able to provide detailed results within the space limitations of this paper.4 5.1 Perplexity computation with 5-gram language models We compared the performance of TPT-encoded language models against three other language model implementations: the SRI language modeling toolkit (Stolcke, 2002), IRSTLM (Federico and Cettolo, 2007), and the language model implementation currently used in the Portage SMT system (Badr et al., 2007), which uses a pointer-based implementation but is able to perform fast LM filtering at load time.


