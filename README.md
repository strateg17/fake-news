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
    
  
