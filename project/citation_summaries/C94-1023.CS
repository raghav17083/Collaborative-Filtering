In addition, explicitly using the left context symbols allows easy use of smoothing techniques, such as deleted interpolation (Bahl, Jelinek, and Mercer 1983), clustering techniques (Brown et al. 1992), and model refinement techniques (Lin, Chiang, and Su 1994) to estimate the probabilities more reliably by changing the window sizes of the context and weighting the various estimates dynamically.


It is possible to use unsupervised learning to train stochastic taggers without the need for a manually annotated corpus by using the Baum-Welch algorithm \[Baum, 1972; Jelinek, 1985; Cutting et al. , 1992; Kupiec, 1992; Elworthy, 1994; Merialdo, 1995\].


Almost all of the work in the area of automatically trained taggers has explored Markov-model based part of speech tagging \[Jelinek, 1985; Church, 1988; Derose, 1988; DeMarcken, 1990; Cutting et al. , 1992; Kupiec, 1992; Charniak et al. , 1993; Weischedel et al. , 1993; Schutze and Singer, 1994; Lin et al. , 1994; Elworthy, 1994; Merialdo, 1995\].


