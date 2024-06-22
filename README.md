![Alt text](./"images/img1.png")



Fake news detection 
======

## Abstract
The proliferation of fake and manipulated news poses a significant threat to the integrity of information dissemination, particularly within the context of Russian media. This paper underscores the critical need for robust assessment frameworks to identify and mitigate the impact of such misinformation. By utilizing a comprehensive dataset, we aim to explore patterns and similarities in fake news narratives propagated by Russian sources. Through clustering techniques, we will analyze the dissemination strategies and thematic consistencies of these false narratives, and compare these clusters with geopolitical regions affected by Russian influence. Such analysis is vital for developing countermeasures and informing policy-making to combat misinformation. Additionally, we will create an interactive visualization tool to map the spread of fake news, providing a dynamic representation of data over various time periods and thematic attributes. This visualization will enhance our understanding of the data and validate our findings. Furthermore, we will evaluate the clusters' relevance by examining their correlation with real-world events and their potential utility in predictive modeling. This study not only highlights the pervasive nature of Russian fake news but also contributes to the development of effective strategies to counteract the dissemination of manipulated information.


## Goal
Inspired by famous [TEDx talk of Simon Sinek about How great leaders inspire action](https://www.ted.com/talks/simon_sinek_how_great_leaders_inspire_action?language=en) we were wondering to answer three question: 

**1. WHY?** 
To master our skills and contribute to developments that strengthen Ukraine’s position in the information war

**2. WHAT?** 
Develop a model capable of identifying fakes in news and social media


**3. HOW?**
Applying modern ML algorithmes to fake and propaganda texts


## Literature review
Analysis of Disinformation and Fake News Detection Using Fine-Tuned Large Language Mode 🔗
Pavlyshenko, B. (2023)

Method and models for sentiment analysis and hidden propaganda finding 🔗
R. Strubytskyi, N. Shakhovska (2023)


Propaganda Detection in Text Data Based on NLP and Machine Learning 🔗
Oliinyk, V., Vysotska, V., Burov, Y., Mykich, K., Basto-Fernandes, V. (2020)

RU22Fact: Optimizing Evidence for Multilingual Explainable Fact-Checking on Russia-Ukraine Conflict 🔗
Yirong Zeng, Xiao Ding, Yi Zhao, Xiangyu Li, Jie Zhang, Chao Yao, Ting Liu, Bing Qin (2024)

## Methodology

### Data Preparation
The dataset used for training and testing the model consists of fact-checked claims, labeled as either 'Supported' (0) or 'Refuted' (1). The data is preprocessed to remove any rows with missing labels, and the labels are mapped to numeric values for compatibility with machine learning algorithms. The dataset is then split into training and testing sets using an 80-20 split ratio to evaluate the model's performance.

### Text Vectorization
Text data is transformed into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization. This method converts the raw text into a matrix of TF-IDF features, which reflects the importance of a word in a document relative to the entire dataset. The `TfidfVectorizer` from the `sklearn` library is used for this purpose.

### Model Training
A LightGBM classifier (`LGBMClassifier`) is trained on the TF-IDF transformed training data. LightGBM is a gradient boosting framework that uses tree-based learning algorithms, known for its efficiency and high performance on large datasets.

### Hyperparameter Optimization
To find the best hyperparameters for the LightGBM model, we use Optuna, a hyperparameter optimization framework. Optuna conducts a series of trials to maximize the model's accuracy on the validation set by tuning parameters such as learning rate, number of leaves, and regularization terms.

### Model Evaluation
The optimized LightGBM model is evaluated on the test set to assess its accuracy. The accuracy scores for both the training and testing sets are calculated and displayed, providing insights into the model's performance and potential overfitting.

### Deployment
The trained LightGBM model and the fitted TF-IDF vectorizer are saved using `joblib`. A Gradio interface is created to deploy the model as a web application, allowing users to input text and receive predictions on whether the text is 'Fake' or 'Not Fake'. The model and vectorizer are loaded in the Gradio app, and the input text is transformed and classified in real-time.

### Code Implementation
The following key Python libraries are used in this implementation:
- `pandas` and `numpy` for data manipulation and numerical operations.
- `scikit-learn` for TF-IDF vectorization, model training, and evaluation.
- `lightgbm` for implementing the LightGBM classifier.
- `optuna` for hyperparameter optimization.
- `joblib` for saving and loading the model and vectorizer.
- `gradio` for creating the web interface to deploy the model.

This methodology ensures a systematic approach to preprocessing, model training, evaluation, and deployment, resulting in an efficient and user-friendly fake news classification tool.


## Proposed datasets
1. [Fake and real news dataset on kaggle.com](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset.)
2. [ Propaganda Diary. Vox Ukraine](https://russiandisinfo.voxukraine.org/en/narratives)
3. [Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london)
4. [RU22Fact: Optimizing Evidence for Multilingual Explainable Fact-Checking on Russia-Ukraine Conflict](https://github.com/zeng-yirong/ru22fact)



## Organisation of the repository
In order to be able to run our notebook, you should have a folder structure similar to:

    .
    ├── data                                      # Folder with data
    ├── models                                    # Folder with models
    ├── gradio                                    # Folder with code for gradio deployment
    ├── notebooks                                 # Folder with all notebooks
    │ ├── data_extractor.ipynb                    # Notebook with extraction data from archieve
    │ ├── data_visualization.ipynb                # Notebook with data EDA and topic modeling
    │ ├── hyperparams_tuning_optuna.ipynb         # Notebook with best model tuning
    │ ├── model_deployment.ipynb                  # Notebook with code for deployment on gradio temporary URL
    │ ├── model_selection.ipynb                   # Notebook with best model selection
    ├── .gitignore                                
    ├── params.txt                                # Best model params
    ├── environment.yml                           # File with ENV for conda
    └── README.md       
    
  
