N04-1026:1	84:192	Lexical information has been shown to improve speech-based emotion prediction in other domains (Litman et al. , 2001; Lee et al. , 2002; Ang et al. , 2002; Batliner et al. , 2003; Devillers et al. , 2003; Shafran et al. , 2003), so our rst non-acoustic-prosodic feature represents the transcription3 of each student turn as a word occurrence vector (indicating the lexical items that are present in the turn).
---------------------------------------------------
N04-1026:2	77:192	Acoustic-Prosodic Features a28 4 normalized fundamental frequency (f0) features: maximum, minimum, mean, standard deviation a28 4 normalized energy (RMS) features: maximum, minimum, mean, standard deviation a28 4 normalized temporal features: total turn duration, duration of pause prior to turn, speaking rate, amount of silence in turn Non-Acoustic-Prosodic Features a28 lexical items in turn a28 6 automatic features: turn begin time, turn end time, isTemporalBarge-in, isTemporalOverlap, #words in turn, #syllables in turn a28 6 manual features: #false starts in turn, isPriorTutorQuestion, isQuestion, isSemanticBarge-in, #canonical expressions in turn, isGrounding Identi er Features: subject, subject gender, problem Figure 2: Features Per Student Turn Following other studies of spontaneous dialogues (Ang et al. , 2002; Lee et al. , 2001; Batliner et al. , 2003; Shafran et al. , 2003), our acoustic-prosodic features represent knowledge of pitch, energy, duration, tempo and pausing.
---------------------------------------------------
N04-1026:3	88:192	The number of words and syllables in a turn provide alternative ways to quantify turn duration (Litman et al. , 2001).
---------------------------------------------------
N04-1026:4	140:192	6 Adding Context-Level Features Research in other domains (Litman et al. , 2001; Batliner et al. , 2003) has shown that features representing the dialogue context can sometimes improve the accuracy of predicting negative user states, compared to the use of features computed from only the turn to be predicted.
---------------------------------------------------
N04-1026:5	18:192	As a result of this mismatch, recent work motivated by spoken dialogue applications has started to use naturally-occurring speech to train emotion predictors (Litman et al. , 2001; Lee et al. , 2001; Ang et al. , 2002; Lee et al. , 2002; Batliner et al. , 2003; Devillers et al. , 2003; Shafran et al. , 2003), but often predicts emotions using only acoustic-prosodic features that would be automatically available to a dialogue system in real-time.
---------------------------------------------------
W01-1610:6	144:150	These experiments show that corrections and aware sites can be classi ed as such automatically, with a considerable degree of accuracy (Litman et al. , 2001;; Hirschberg et al. , 2001).
---------------------------------------------------
W01-1610:7	131:150	In terms of distinguishing features which might explain or help to identify these turns, we have previously examined the acoustic and prosodic features of aware sites (Litman et al. , 2001).
---------------------------------------------------
W01-1610:8	15:150	In other papers (Swerts et al. , 2000;; Hirschberg et al. , 2001;; Litman et al. , 2001), we have already given some descriptive statistics on corrections and aware sites and we have been looking at methods to automatically predict these two utterance categories.
---------------------------------------------------
N03-2018:9	7:50	Although (Ang et al. , 2002; Litman et al. , 2001; Batliner et al. , 2000) have hand-labeled naturally-occurring utterances in a variety of corpora for various emotions, then extracted acoustic, prosodic and lexical features and used machine-learning techniques to develop predictive models, little work to date has addressed emotion detection in computer-based educational settings.
---------------------------------------------------
W03-0205:10	22:187	Other research projects (Mostow and Aist, 2001; Fry et al. , 2001) have shown that basic spoken natural language capabilities can be implemented quite effectively in computer tutoring systems.
---------------------------------------------------
W03-0205:11	23:187	Moreover, speech contains prosodic and acoustic information which has been shown to improve the accuracy of predicting emotional states (Ang et al. , 2002; Batliner et al. , 2000) and user responses to system errors (Litman et al. , 2001) that are useful for triggering system adaptation.
---------------------------------------------------
W03-0409:12	110:233	For more information on our tasks and features, see (Litman et al. , 2000; Hirschberg et al. , 2001; Litman et al. , 2001).
---------------------------------------------------
