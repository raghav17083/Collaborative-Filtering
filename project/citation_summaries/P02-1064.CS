In our system, we used a pool-based active learning framework that is similar as Manabu Sassano (2002) used, this is shown in figure 1.


(2006), who combined margin scores with corpus diversity, and Sassano (2002), who considered SVM active learning 235 100150200250300350400450500Labels0.65 0.700.75 0.800.85 0.900.95 Test Accuracy 20 Newsgroups PA Random (82.53)CW Random (92.92)PA Margin (88.06) CW Margin (95.39)ACL (95.51) 100150200250300350400450500Labels 0.75 0.80 0.85 0.90 Test Accuracy All PA Random (81.30)CW Random (86.67)PA Margin (83.99) CW Margin (88.61)ACL (88.79) 0.20.40.60.81.01.21.4ACL Labels0.2 0.4 0.6 0.8 1.0 1.2 1.4 CW Margin Labels Reuters20 NewsgroupsSentiment Spam Figure 1: Results averaged over 20 Newsgroups (left) and all datasets (center) showing test accuracy over active learning rounds.


In order to overcome this, several methods are proposed, including minimally-supervised learning methods (e.g. , (Yarowsky, 1995; Blum and Mitchell, 1998)), and active learning methods (e.g. , (Thompson et al. , 1999; Sassano, 2002)).


