# MoodMentor
Mental Health Prediction

Objective:=
In today's digital landscape, social media platforms predominantly rely on engagement-based algorithms, which often reinforce repetitive content that may not support users' mental well-being. This approach overlooks the potential to use technology for fostering positive mental health. There is a pressing need for tools that not only detect mental health sentiments but also recommend tailored content to enhance users' well-being.

Scope:-
The Mental Health Prediction System addresses this challenge by leveraging machine learning and NLP to analyze user sentiments in real time. It offers the potential to integrate with social media platforms, enabling feeds to prioritize supportive and mindful content, creating a healthier digital environment.

About dataset:-
This comprehensive dataset is a meticulously curated collection of mental health statuses tagged from various statements. The dataset amalgamates raw data from multiple sources, cleaned and compiled to create a robust resource for developing chatbots and performing sentiment analysis.

Features:
  1. unique_id: A unique identifier for each entry.
  2. Statement: The textual data or post.  
  3. Mental Health Status: The tagged mental health status of the statement.

This dataset was created by aggregating and cleaning data from various publicly available datasets on Kaggle. Special thanks to the original dataset creators for their contributions.

Mthodology:-
Split : Splitting the data into training and testing sets. The training set helps the model learn patterns in the data, while the testing set is used to evaluate the model's performance on unseen data.
The cleaning of dataset involves following sequence of steps.
  Began by thoroughly cleaning the data to ensure its quality and consistency. This process included removing any blank rows to prevent data gaps and converting all text to lowercase for uniformity. We also     
  removed punctuation and numerical characters, which are often unnecessary for text analysis. Short forms and contractions, such as "’ll" (short for "will"), were expanded for clarity . Additionally, we   
  applied stemming and tokenization techniques to break down the text into individual root words and tokens, respectively, enhancing the data's readiness for further analysis. 
 
	Data Preprocessing : This stage includes several essential steps to prepare the data for model training. First, we applied label encoding to the target variable, assigning numerical labels to each category 
  (e.g., 0 for anxiety, 1 for depression, and so on). This encoding enables the model to interpret categorical labels as numerical values. Next, we performed vectorization to convert words into numerical 
  representations, facilitating the model’s ability to process and analyze textual data [2]. Specifically, we created a feature set consisting of the top 50,000 words, capturing the most relevant terms for 
  building an accurate and comprehensive model. This extensive feature set helps the model capture nuances in the text related to different mental health categories.

  Mathematical expression of TF-IDF, for a word or term w occuring in a document D is given by,
    TF-IDF(w,D)=TF(w,D)*IDF(w)
	Implementing ML models : Employed various classifiers, including Naive Bayes (Multinomial), Logistic Regression with a multiclass approach, Decision Tree, Random Forest, and ensemble methods like bagging and      boosting, to enable the model to learn effectively from the data. Hyperparameter tuning was performed to optimize model performance. Since the dataset is imbalanced, we applied random sampling             
  techniques to address the imbalance and ensure the models can learn from all classes more effectively. 
	
 Evaluation : Accuracy, F1-score and ROC-AUC Score are considered as the parameter to selct the model while ensurung the confusion metrics is also optimal.
 1. Accuracy measures how close the results are to the true or expected values. It is a fundamental paramter used to measure the correctness of predictions. Its formula is given by,
           Accuracy=(TP+TN)/(TP+FP+TN+FN)
 2. F1-score combines both recall and precision and is considered a better representative of a model’s performance. It is the harmonic mean of precision and recall. It is calculated using,
           F1-score=2×(Precison×Recall)/(Precision+Recall)
 3. The ROC-AUC score measures the model's ability to distinguish between classes by plotting the trade-off between the true positive rate (TPR) and false positive rate (FPR) at various thresholds. A higher score 
    (closer to 1) indicates better classification performance, with 0.5 representing random guessing.

Choosing XGBoost classifier as the best among all others.


