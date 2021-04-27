Because the set of target sentences T considered can be huge, previous work (Fung and Cheung, 2004; Resnik and Smith, 2003; Snover et al., 2008; Munteanu and Marcu, 2005) pre-selects target sentences T at the document level . We have re-implemented a particular filtering scheme based on BM25 (Quirk et al., 2007; Utiyama and Isahara, 2003; Robertson et al., 1995).


# cands reports the size of the overall search space in terms of sentence pairs processed . Data Source # cands # pairs Bleu Baseline 1.826 M 42.3 + Giga 999.3 B 1.357 M 45.7 + Giga (BM25) 25.4 B 0.609 M 44.8 Baseline 2.222 M 45.3 + News Data 2006 77.8 B 56 K 47.2 5 Future Work and Discussion In this paper, we have presented a novel beamsearch algorithm to extract sentence pairs from comparable data . It can avoid any pre-filtering at the document level (Resnik and Smith, 2003; Snover et al., 2008; Utiyama and Isahara, 2003; Munteanu and Marcu, 2005; Fung and Cheung, 2004).


It differs from similar algorithms that select translation correspondences explicitly at the document level (Fung and Cheung, 2004; Resnik and Smith, 2003; Snover et al., 2008; Munteanu and Marcu, 2005; Quirk et al., 2007; Utiyama and Isahara, 2003).


With the help of an efficient implementation, it avoids any translation candidate selection at the document level (Resnik and Smith, 2003; Smith, 2002; Snover et al., 2008; Utiyama and Isahara, 2003; Munteanu and Marcu, 2005; Fung and Cheung, 2004).


