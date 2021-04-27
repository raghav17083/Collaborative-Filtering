We conducted an experimental evaluation using automatic summarization evaluation data developed for TSC-3 (Hirao et al. , 2004a).


Our method is based on the Extended String Subsequence Kernel (ESK) (Hirao et al. , 2004b) which is a kind of convolution kernel (Collins and Duffy, 2001).


Extended String Subsequence Kernel (ESSK) Formally, ESSK is defined as follows (Hirao et al., 2004): Kessk(T,U) = dsummationdisplay m=1 summationdisplay tiT summationdisplay ujU Km(ti,uj) Km(ti,uj) = braceleftbigg val(t i,uj) if m = 1 Kprimem1(ti,uj)  val(ti,uj) Here, Kprimem(ti,uj) is defined below.


(Hirao et al., 2004) represent the sentences using Dependency Tree Path (DTP) to incorporate syntactic information.


Some improvements on BOW are given by the use of dependency trees and syntactic parse trees (Hirao et al., 2004), (Punyakanok et al., 2004), (Zhang and Lee, 2003), but these, too are not adequate when dealing with complex questions whose answers are expressed by long and articulated sentences or even paragraphs.


