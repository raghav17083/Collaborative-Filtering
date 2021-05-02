import pandas as pd
import numpy as np
import streamlit as st


st.sidebar.title("Researh Paper Recommendation System")
analysis = st.sidebar.selectbox('Select an Option',['Get recommendations','About'])

if analysis =='Get recommendations':
	# print("HERE")
	user_input = st.text_area("Type the name of the paper to get recommendations", "Enter Title")
	# POI_ID = "P12-1041"
	print(user_input)

	menu = ["User-Item","Co-Cited and Co-Referenced Based Similarity","KMeans","Kernel K Means","Subspace Clustering","Latent Factor Model","Non Negative Matrix Factorisation","Binary matrix Factorisation","CCIDF- Co-citation Inverse Document Frequency"]
	choice = st.selectbox("Select an Algorithm ",menu)

	if(choice == "KMeans"):
		submenu = ["Eucledian","Cosine","Jaccard"]
		distance_metric = st.selectbox("Select a distance metric",submenu)

		if(distance_metric =="Eucledian"):

			if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. Extracting Bacteria Biotopes with Semi-supervised Named Entity Recognition and Coreference Resolution  ")
				st.markdown("2. A Clustering Approach For Unsupervised Chinese Coreference Resolution  ")
				st.markdown("3. Extending English ACE 2005 Corpus Annotation with Ground-truth Links to Wikipedia")
				st.markdown("4. The Impact Of Morphological Stemming On Arabic Mention Detection And Coreference Resolution ")
				st.markdown("5. A Chain-starting Classifier of Definite NPs in Spanish ")

				st.image("Result_images/KMeansP12-1041.png",width = 500)

			else : 

				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. POS Tagging Versus Classes In Language Modeling ")
				st.markdown("2. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")
				st.markdown("3. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("4. Discourse-New Detectors For Definite Description Resolution: A Survey And A Preliminary Proposal   ")
				st.markdown("5. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")

				st.image("Result_images/KmeansP10-1142.png",width = 500)


		if(distance_metric =="Cosine"):

			if(user_input=="Coreference Semantics from Web Features"):

				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Chain-starting Classifier of Definite NPs in Spanish ")
				st.markdown("2. Extracting Bacteria Biotopes with Semi-supervised Named Entity Recognition and Coreference Resolution  ")
				st.markdown("3. Extending English ACE 2005 Corpus Annotation with Ground-truth Links to Wikipedia")
				st.markdown("4. A Clustering Approach For Unsupervised Chinese Coreference Resolution  ")
				st.markdown("5. The Impact Of Morphological Stemming On Arabic Mention Detection And Coreference Resolution ")
				
				st.image("Result_images/Kmeans_CosineP12-1041.png",width = 500)

			else : 

				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Discourse-New Detectors For Definite Description Resolution: A Survey And A Preliminary Proposal   ")
				st.markdown("2. POS Tagging Versus Classes In Language Modeling ")
				st.markdown("3. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("4. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")
				st.markdown("5. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")


				st.image("Result_images/Kmeans_cosineP10-1142.png",width = 500)

		if(distance_metric =="Jaccard"):

			if(user_input=="Coreference Semantics from Web Features"):

				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Chain-starting Classifier of Definite NPs in Spanish ")
				st.markdown("2. Extending English ACE 2005 Corpus Annotation with Ground-truth Links to Wikipedia")
				st.markdown("3. A Clustering Approach For Unsupervised Chinese Coreference Resolution  ")
				st.markdown("4. Extracting Bacteria Biotopes with Semi-supervised Named Entity Recognition and Coreference Resolution  ")
				st.markdown("5. The Impact Of Morphological Stemming On Arabic Mention Detection And Coreference Resolution ")
				
				st.image("Result_images/Kmeans_jaccardP12-1041.png",width = 500)

			else : 

				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Discourse-New Detectors For Definite Description Resolution: A Survey And A Preliminary Proposal   ")
				st.markdown("2. POS Tagging Versus Classes In Language Modeling ")
				st.markdown("3. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")
				st.markdown("4. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("5. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")
				


				st.image("Result_images/KmeansP10-1142.png",width = 500)




	if(choice=="User-Item"):

		if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Study of Information Retrieval Weighting Schemes for Sentiment Analysis  ")
				st.markdown("2. Using Emoticons To Reduce Dependency In Machine Learning Techniques For Sentiment Classification ")
				st.markdown("3. Sentiment Classification using Rough Set based Hybrid Feature Selection")
				st.markdown("4. Mine the Easy Classify the Hard: A Semi-Supervised Approach to Automatic Sentiment Classification  ")
				st.markdown("5. Coreference Resolution across Corpora: Languages Coding Schemes and Preprocessing Information ")

				st.image("Result_images/userbased_P10-1141.png",width = 500)
		else:


				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Supervised Noun Phrase Coreference Research: The First Fifteen Years ")
				st.markdown("2. Supervised Models for Coreference Resolution ")
				st.markdown("3. Graph-Cut-Based Anaphoricity Determination for Coreference Resolution")
				st.markdown("4. Unsupervised Models for Coreference Resolution ")
				st.markdown("5. Assigning Polarity Scores to Reviews Using Machine Learning Techniques")
				
				st.image("Result_images/userbased_P10-1142.png",width = 500)
		


	if(choice == "Kernel K Means"):
		kernelList = ['linear','rbf','polynomial','laplacian','sigmoid']
		kernel = st.selectbox("Select a Kernel",kernelList)

		if(kernel=='linear'):
			if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. The Tradeoffs Between Open and Traditional Relation Extraction  ")
				st.markdown("2. A Study of Information Retrieval Weighting Schemes for Sentiment Analysis  ")
				st.markdown("3. First-Order Probabilistic Models for Coreference Resolution ")
				st.markdown("4. Understanding the Value of Features for Coreference Resolution  ")
				st.markdown("5. Coreference Resolution across Corpora: Languages Coding Schemes and Preprocessing Information ")

				st.image("Result_images/PaperAlgoP10-1142.png",width = 500)


			else:


				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Unsupervised Learning of Narrative Event Chains")
				st.markdown("2. An Empirically Based System For Processing Definite Descriptions ")
				st.markdown("3. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("4. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")
				st.markdown("5. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")

				st.image("Result_images/PaperAlgoP10-1142.png",width = 500)



		if(kernel=='rbf'):
			
			if(user_input=="Coreference Semantics from Web Features"):

				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Chain-starting Classifier of Definite NPs in Spanish ")
				st.markdown("2. Extending English ACE 2005 Corpus Annotation with Ground-truth Links to Wikipedia")
				st.markdown("3. A Clustering Approach For Unsupervised Chinese Coreference Resolution  ")
				st.markdown("4. Extracting Bacteria Biotopes with Semi-supervised Named Entity Recognition and Coreference Resolution  ")
				st.markdown("5. The Impact Of Morphological Stemming On Arabic Mention Detection And Coreference Resolution ")
				
				st.image("Result_images/Kmeans_jaccardP12-1041.png",width = 500)

			else : 

				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Discourse-New Detectors For Definite Description Resolution: A Survey And A Preliminary Proposal   ")
				st.markdown("2. POS Tagging Versus Classes In Language Modeling ")
				st.markdown("3. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")
				st.markdown("4. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("5. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")
				


				st.image("Result_images/KmeansP10-1142.png",width = 500)

		if(kernel=='polynomial'):
			
			if(user_input=="Coreference Semantics from Web Features"):

				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Chain-starting Classifier of Definite NPs in Spanish ")
				st.markdown("2. Extracting Bacteria Biotopes with Semi-supervised Named Entity Recognition and Coreference Resolution  ")
				st.markdown("3. Extending English ACE 2005 Corpus Annotation with Ground-truth Links to Wikipedia")
				st.markdown("4. A Clustering Approach For Unsupervised Chinese Coreference Resolution  ")
				st.markdown("5. The Impact Of Morphological Stemming On Arabic Mention Detection And Coreference Resolution ")
				
				st.image("Result_images/Kmeans_CosineP12-1041.png",width = 500)

			else : 

				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Discourse-New Detectors For Definite Description Resolution: A Survey And A Preliminary Proposal   ")
				st.markdown("2. POS Tagging Versus Classes In Language Modeling ")
				st.markdown("3. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("4. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")
				st.markdown("5. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")


				st.image("Result_images/Kmeans_cosineP10-1142.png",width = 500)

		if(kernel=='laplacian'):
			if(user_input=="Coreference Semantics from Web Features"):

				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Chain-starting Classifier of Definite NPs in Spanish ")
				st.markdown("2. Extending English ACE 2005 Corpus Annotation with Ground-truth Links to Wikipedia")
				st.markdown("3. A Clustering Approach For Unsupervised Chinese Coreference Resolution  ")
				st.markdown("4. Extracting Bacteria Biotopes with Semi-supervised Named Entity Recognition and Coreference Resolution  ")
				st.markdown("5. The Impact Of Morphological Stemming On Arabic Mention Detection And Coreference Resolution ")
				
				st.image("Result_images/Kmeans_jaccardP12-1041.png",width = 500)

			else : 

				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Discourse-New Detectors For Definite Description Resolution: A Survey And A Preliminary Proposal   ")
				st.markdown("2. POS Tagging Versus Classes In Language Modeling ")
				st.markdown("3. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")
				st.markdown("4. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("5. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")
				


				st.image("Result_images/KmeansP10-1142.png",width = 500)

		if(kernel=='sigmoid'):
			
			if(user_input=="Coreference Semantics from Web Features"):

				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Chain-starting Classifier of Definite NPs in Spanish ")
				st.markdown("2. Extracting Bacteria Biotopes with Semi-supervised Named Entity Recognition and Coreference Resolution  ")
				st.markdown("3. Extending English ACE 2005 Corpus Annotation with Ground-truth Links to Wikipedia")
				st.markdown("4. A Clustering Approach For Unsupervised Chinese Coreference Resolution  ")
				st.markdown("5. The Impact Of Morphological Stemming On Arabic Mention Detection And Coreference Resolution ")
				
				st.image("Result_images/Kmeans_CosineP12-1041.png",width = 500)

			else : 

				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Discourse-New Detectors For Definite Description Resolution: A Survey And A Preliminary Proposal   ")
				st.markdown("2. POS Tagging Versus Classes In Language Modeling ")
				st.markdown("3. Automatic Distinction Of Arguments And Modifiers: The Case Of Prepositional Phrases")
				st.markdown("4. Extracting And Evaluating General World Knowledge From The Brown Corpus  ")
				st.markdown("5. Multilingual Aligned Parallel Treebank Corpus Reflecting Contextual Information And Its Applications ")


				st.image("Result_images/Kmeans_cosineP10-1142.png",width = 500)



	if(choice=="Co-Cited and Co-Referenced Based Similarity"):

		if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. The Tradeoffs Between Open and Traditional Relation Extraction  ")
				st.markdown("2. An Empirically Based System For Processing Definite Descriptions  ")
				st.markdown("3. First-Order Probabilistic Models for Coreference Resolution ")
				st.markdown("4. Error-Driven Analysis of Challenges in Coreference Resolution ")
				st.markdown("5. Coreference Resolution across Corpora: Languages Coding Schemes and Preprocessing Information ")

				st.image("Result_images/PaperAlgoP12-1041.png",width = 500)
		else:


				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Unsupervised Learning of Narrative Event Chains")
				st.markdown("2. An Empirically Based System For Processing Definite Descriptions ")
				st.markdown("3. CogNIAC: High Precision Coreference With Limited Knowledge And Linguistic Resources")
				st.markdown("4. Error-Driven Analysis of Challenges in Coreference Resolution ")
				st.markdown("5. Coreference Resolution across Corpora: Languages Coding Schemes and Preprocessing Information ")
				
				st.image("Result_images/PaperAlgoP10-1142.png",width = 500)

	if(choice=="CCIDF- Co-citation Inverse Document Frequency"):

		if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Large-Scale Exploration Of Effective Global Features For A Joint Entity Detection And Tracking Model  ")
				st.markdown("2. Coreference Resolution across Corpora: Languages Coding Schemes and Preprocessing Information  ")
				st.markdown("3. Creating Robust Supervised Classifiers via Web-Scale N-Gram Data  ")
				st.markdown("4. Understanding the Value of Features for Coreference Resolution  ")
				st.markdown("5. Conundrums in Noun Phrase Coreference Resolution: Making Sense of the State-of-the-Art")

				st.image("Result_images/ccidfP12-1041.png",width = 500)
		else:


			st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
			st.markdown("1. Improving Pronoun Resolution Using Statistics-Based Semantic Compatibility Information ")
			st.markdown("2. Capturing Salience with a Trainable Cache Model for Zero-anaphora Resolution ")
			st.markdown("3. Accurate Semantic Class Classifier for Coreference Resolution")
			st.markdown("4. Competitive Self-Trained Pronoun Interpretation")
			st.markdown("5. Corpus-Based Learning For Noun Phrase Coreference Resolution")
			
			st.image("Result_images/ccidfP10-1142.png",width = 500)


	if(choice=="Latent Factor Model"):

		if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. Automatic Acquisition Of Hyponyms From Large Text Corpora ")
				st.markdown("2. A Machine Learning Approach To Coreference Resolution Of Noun Phrases ")
				st.markdown("3. Automatic Retrieval and Clustering of Similar Words   ")
				st.markdown("4. Improving Machine Learning Approaches To Coreference Resolution ")
				st.markdown("5. A Model-Theoretic Coreference Scoring Scheme    ")

				st.image("Result_images/nnmf.png",width = 500)
		else:


			st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
			st.markdown("1. Attention Intentions And The Structure Of Discourse ")
			st.markdown("2. A Machine Learning Approach To Coreference Resolution Of Noun Phrases   ")
			st.markdown("3. Improving Machine Learning Approaches To Coreference Resolution  ")
			st.markdown("4. A Model-Theoretic Coreference Scoring Scheme")
			st.markdown("5. Centering: A Framework For Modeling The Local Coherence Of Discourse ")
			
			st.image("Result_images/nnmf.png",width = 500)


	if(choice=="Non Negative Matrix Factorisation"):

		if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. Thumbs Up Or Thumbs Down? Semantic Orientation Applied To Unsupervised Classification Of Reviews  ")
				st.markdown("2. Discriminative Training Methods For Hidden Markov Models: Theory And Experiments With Perceptron Algorithms   ")
				st.markdown("3. Recognizing Contextual Polarity In Phrase-Level Sentiment Analysis")
				st.markdown("4. Coarse-To-Fine N-Best Parsing And MaxEnt Discriminative Reranking  ")
				st.markdown("5. Online Large-Margin Training Of Dependency Parsers  ")

				st.image("Result_images/nnfm_final.png",width = 500)
		else:


				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Thumbs Up? Sentiment Classification Using Machine Learning Techniques ")
				st.markdown("2. A Maximum-Entropy-Inspired Parser ")
				st.markdown("3. Three Generative Lexicalized Models For Statistical Parsing")
				st.markdown("4. Recognizing Contextual Polarity In Phrase-Level Sentiment Analysis ")
				st.markdown("5. The Mathematics Of Statistical Machine Translation: Parameter Estimation ")
				
				st.image("Result_images/nnfm_final.png",width = 500)


	if(choice=="Subspace Clustering"):

		if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. Discriminative Reordering with Chinese Grammatical Relations Features     ")
				st.markdown("2. Non-Classical Lexical Semantic Relations ")
				st.markdown("3. A System For Extraction Of Temporal Expressions From French Texts Based On Syntactic And Semantic Constraints")
				st.markdown("4. Bootstrapping Parallel Treebanks ")
				st.markdown("5. Development Of The Concept Dictionary Implementation Of Lexical Knowledge   ")

				st.image("Result_images/Subspace_eucledianP12-1041.png",width = 500)
		else:


				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Thumbs Up? Sentiment Classification Using Machine Learning Techniques ")
				st.markdown("2. A Maximum-Entropy-Inspired Parser ")
				st.markdown("3. Three Generative Lexicalized Models For Statistical Parsing")
				st.markdown("4. Recognizing Contextual Polarity In Phrase-Level Sentiment Analysis ")
				st.markdown("5. The Mathematics Of Statistical Machine Translation: Parameter Estimation ")
				
				st.image("Result_images/PaperAlgoP10-1142.png",width = 500)

	if(choice=="Binary matrix Factorisation"):

		if(user_input=="Coreference Semantics from Web Features"):
				st.subheader("Papers Recommended for Paper Title- Coreference Semantics from Web Features")
				st.markdown("1. A Study of Information Retrieval Weighting Schemes for Sentiment Analysis  ")
				st.markdown("2. Using Emoticons To Reduce Dependency In Machine Learning Techniques For Sentiment Classification ")
				st.markdown("3. Sentiment Classification using Rough Set based Hybrid Feature Selection")
				st.markdown("4. Mine the Easy Classify the Hard: A Semi-Supervised Approach to Automatic Sentiment Classification  ")
				st.markdown("5. Coreference Resolution across Corpora: Languages Coding Schemes and Preprocessing Information ")

				st.image("Result_images/BMF.png",width = 500)
		else:


				st.subheader("Papers Recommended for Paper Title- Supervised Noun Phrase Coreference Research: The First Fifteen Years")
				st.markdown("1. Building A Large Annotated Corpus Of English: The Penn Treebank   ")
				st.markdown("2. The Mathematics Of Statistical Machine Translation: Parameter Estimation  ")
				st.markdown("3. Minimum Error Rate Training In Statistical Machine Translation ")
				st.markdown("4. A Systematic Comparison Of Various Statistical Alignment Models  ")
				st.markdown("5. Moses: Open Source Toolkit for Statistical Machine Translation ")
				
				st.image("Result_images/BMF.png",width = 500)
		
	

		
	# print(POI_ID)

