1 Introduction (Chen et al., 2007) describes an architecture that allows to combine statistical machine translation (SMT) with one or multiple rule-based machine translation (RBMT) systems in a multi-engine setup.


Previous work on building hybrid systems includes, among others, approaches using reranking, regeneration with an SMT decoder (Eisele et al., 2008; Chen et al., 2007), and confusion networks (Matusov et al., 2006; Rosti et al., 2007; He et al., 2008).


c2009 Association for Computational Linguistics Combining Multi-Engine Translations with Moses Yu Chen1, Michael Jellinghaus1, Andreas Eisele1,2,Yi Zhang1,2, Sabine Hunsicker1, Silke Theison1, Christian Federmann2, Hans Uszkoreit1,2 1: Universitat des Saarlandes, Saarbrucken, Germany 2: Deutsches Forschungszentrum fur Kunstliche Intelligenz GmbH, Saarbrucken, Germany {yuchen,micha,yzhang,sabineh,sith}@coli.uni-saarland.de {eisele,cfedermann,uszkoreit}@dfki.de Abstract We present a simple method for generating translations with the Moses toolkit (Koehn et al., 2007) from existing hypotheses produced by other translation engines.


In contrast to the prior approach presented in (Chen et al., 2007) and (Eisele et al., 2008) which concatenates the phrase tables and adds new features as system markers, our extension method avoids duplicate entries in the final combined table.


138 ID Participant cmu-uka Carnegie Mellon University, USA (Paulik et al. , 2007) cmu-syntax Carnegie Mellon University, USA (Zollmann et al. , 2007) cu Charles University, Czech Republic (Bojar, 2007) limsi LIMSI-CNRS, France (Schwenk, 2007) liu University of Linkoping, Sweden(Holmqvist et al. , 2007) nrc National Research Council, Canada (Ueffing et al. , 2007) pct a commercial MT provider from the Czech Republic saar Saarland University & DFKI, Germany (Chen et al. , 2007) systran SYSTRAN, France & U. Edinburgh, UK (Dugast et al. , 2007) systran-nrc National Research Council, Canada (Simard et al. , 2007) ucb University of California at Berkeley, USA (Nakov and Hearst, 2007) uedin University of Edinburgh, UK (Koehn and Schroeder, 2007) umd University of Maryland, USA (Dyer, 2007) upc University of Catalonia, Spain (Costa-Juss`a and Fonollosa, 2007) upv University of Valencia, Spain (Civera and Juan, 2007) Table 1: Participants in the shared task.


