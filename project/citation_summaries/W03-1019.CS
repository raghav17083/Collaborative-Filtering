This point-wise discriminant function is different from that described in (Kakade et al. , 2002; Altun et al. , 2003), which is calculated based on marginals.


We need to define a segment-wise loss, in contrast to the standard CRF loss, which is sometimes referred to as an (entire) sequential loss (Kakade et al. , 2002; Altun et al. , 2003).


6 Related Work Various loss functions have been proposed for designing CRFs (Kakade et al. , 2002; Altun et al. , 2003).


joint likelihood (JL) productdisplay i p parenleftBig xi,yi | vector parenrightBig conditional likelihood (CL) productdisplay i p parenleftBig yi | xi,vector parenrightBig classification accuracy (Juang and Katagiri, 1992) summationdisplay i (yi, y(xi)) expected classification accuracy (Klein and Manning, 2002) summationdisplay i p parenleftBig yi | xi,vector parenrightBig negated boosting loss (Collins, 2000)  summationdisplay i p parenleftBig yi | xi,vector parenrightBig1 margin (Crammer and Singer, 2001)  s.t. bardbl vectorbardbl  1;i,y negationslash= yi, vector  (vectorf(xi,yi )  vectorf(xi,y))   expected local accuracy (Altun et al. , 2003) productdisplay i productdisplay j p parenleftBig lscriptj(Y ) = lscriptj(yi ) | xi,vector parenrightBig Table 1: Various supervised training criteria.


