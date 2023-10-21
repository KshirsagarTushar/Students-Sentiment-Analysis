# Unveiling Student Sentiments: A Feedback Analysis Study
This project is based on the sentiment analysis of students feedback, where the feedbacks are given by the students of a particular university. Various machine learning algorithms with TF-IDF technique are used, out of which one best model is chosed for predictive modelling.

# Sections
- Overvview
- Dataset Information
- Methodology
- Installation
- How to run
- Authors
- Relevant Links

# Overview
This project has provided valuable insights into the complex realm of student sentiment within the educational context.Through a comprehensive exploration of descriptive and predictive statistical techniques,this study has shed light on various aspects of student sentiment,paving the way for data-driven decision-making in educational institutions.

# Dataset Information
The study uses and compares various different methods for sentiment analysis of student's feedback (a multi-class classification problem). The dataset is expected to be a csv file of type id,polarity,comment where the id is a unique integer identifying the comment (student's feedback), polarity is either 1 (positive) or 2 (negative) or 0 (neutral) and comment is the feedback given by students enclosed in "". Please note that csv headers are not expected and should be removed from the dataset.

# Methodology
- Data Collection:
  Commenced the project by gathering data related to student sentiment. This data is collected through various channels such as student surveys, feedback forms, online forums, and written comments.          Ensured that the data encompasses a wide range of sentiment expressions.
  
- Data Preprocessing:
  Prepared the collected data for analysis by conducting data preprocessing. This phase involves cleaning the data, handling missing values, removal of URLs and HTML links, lower casing, removal of          numbers, removal of stopwords and punctuations, removal of extra white space, lemmatization, standardizing or encoding categorical variables to ensure data quality and consistency.

- Descriptive Sentiment Analysis:
  Performed descriptive sentiment analysis to gain an initial understanding of student sentiment. This analysis includes:
  - Categorizing sentiments into positive, negative, or neutral classes.
  - Generating summary statistics, including sentiment distribution and frequency.
  - Visualizing sentiment trends.

- Predictive Modeling:
  Developed predictive models using machine learning techniques to forecast student sentiment. Steps includes:
  - Feature selection: Choosed relevant variables and features for the predictive model.
  - Model selection: Experimented with various machine learning algorithms with TF-IDF technique(e.g. SVM, Naive Bayes, Random Forest, Ada Boost, Gradient Boosting) to build predictive models.
  - Trained and validated models using historical sentiment data.

- Model Evaluation:
  Evaluated the predictive models using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score. Ensured that the models generalized well to new data to make accurate predictions.

- Documentation and Reporting:
  Documented the entire methodology and results comprehensively in a project report. Included clear explanations of data preprocessing steps, descriptive and predictive analysis techniques, model details,   and actionable recommendations. Utilized visualizations and tables to enhance data interpretation.

  By following this methodology, the project aims to provide a holistic view of student sentiment, from descriptive insights to predictive capabilities, and to equip educational institutions with valuable   information to enhance the student experience and foster a more responsive and adaptive educational environment.
  
# Installation
There is a general requirement of libraries for the project and some of which are specific to individual methods. The required libraries are as follows:
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scipy
- scikit-learn

These libraries can be installed by using the pip installer.

# How to run
For the sake of simplicity I have added everything in a single Python Notebook file. Follow the Notebook, each cell in the notebook is well commented which will help to understand the project steps.

# Author
[Tushar Kshirsagar](https://github.com/KshirsagarTushar)

# Relevant Links
LinkedIn : www.linkedin.com/in/tushar-kshirsagar-8459bb245

GitHub : https://github.com/KshirsagarTushar
