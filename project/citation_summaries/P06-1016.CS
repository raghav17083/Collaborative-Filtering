Zhou et al (2006) further improved the performance by exploring the commonality among related classes in a class hierarchy using hierarchical learning strategy.


Since then, many methods, such as feature-based (Kambhatla 2004; Zhou et al 2005, 2006), tree kernel-based (Zelenko et al 2003; Culotta and Sorensen 2004; Bunescu and Mooney 2005a; Zhang et al 2006) and composite kernel-based (Zhao and Gris hman 2005; Zhang et al 2006), have been proposed in literature.


As the state-of-the-art, Zhang et al (2006) applied the convolution tree kernel (Collins and Duffy 2001) and achieved comparable performance with a state-of-theart linear kernel (Zhou et al 2005) on the 5 relation types in the ACE RDC 2003 corpus.


Composite Kernel In this paper, a composite kernel via polynomial interpolation, as described Zhang et al (2006), is applied to integrate the proposed context-sensitive convolution tree kernel with a state-of-the-art linear kernel (Zhou et al 2005) 7: ),()1(),(),(1 += CPL KKK aa (5) Here, ),( LK and ),( CK indicates the normalized linear kernel and context-sensitive convolution tree kernel respectively while (,)pK  is the polynomial expansion of (,)K  with degree d=2, i.e. 2(,)((,)1)pKK =+ and a is the coefficient (a is set to 0.3 using cross-validation).


System P(%) R(%) F Linear Kernel 78.2 (77.2) 63.4 (60.7) 70.1 (68.0) Context-Sensitive Convolution Tree Kernel 81.1 (80.1) 66.7 (63.8) 73.2 (71.0) Composite Kernel 82.2 (80.8) 70.2 (68.4) 75.8 (74.1) Table 3: Performance of the compos ite kernel via polynomial interpolation on the major relation types of the ACE RDC 2003 (inside the parentheses) and 2004 (outside the parentheses) corpora Comparison with Other Systems ACE RDC 2003 P(%) R(%) F Ours: composite kernel 80.8 (65.2) 68.4 (54.9) 74.1 (59.6) Zhang et al (2006): composite kernel 77.3 (64.9) 65.6 (51.2) 70.9 (57.2) Ours: context-sensitive convolution tree kernel 80.1 (63.4) 63.8 (51.9) 71.0 (57.1) Zhang et al (2006): convolution tree kernel 76.1 (62.4) 62.6 (48.5) 68.7 (54.6) Bunescu et al (2005): shortest path dependency kernel 65.5 (-) 43.8 (-) 52.5 (-) Culotta et al (2004): dependency kernel 67.1 (-) 35.0 (-) 45.8 (-) Zhou et al.


