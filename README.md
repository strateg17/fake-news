Fake news detection 
======

## Abstract
The proliferation of fake and manipulated news poses a significant threat to the integrity of information dissemination, particularly within the context of Russian media. This paper underscores the critical need for robust assessment frameworks to identify and mitigate the impact of such misinformation. By utilizing a comprehensive dataset, we aim to explore patterns and similarities in fake news narratives propagated by Russian sources. Through clustering techniques, we will analyze the dissemination strategies and thematic consistencies of these false narratives, and compare these clusters with geopolitical regions affected by Russian influence. Such analysis is vital for developing countermeasures and informing policy-making to combat misinformation. Additionally, we will create an interactive visualization tool to map the spread of fake news, providing a dynamic representation of data over various time periods and thematic attributes. This visualization will enhance our understanding of the data and validate our findings. Furthermore, we will evaluate the clusters' relevance by examining their correlation with real-world events and their potential utility in predictive modeling. This study not only highlights the pervasive nature of Russian fake news but also contributes to the development of effective strategies to counteract the dissemination of manipulated information.


## Goal
Inspired by famous [TEDx talk of Simon Sinek about How great leaders inspire action](https://www.ted.com/talks/simon_sinek_how_great_leaders_inspire_action?language=en) we were wondering to answer three question: 

**1. WHY?** 
TODO: 

**2. WHAT?** 
- Fake news classification to build a news filters or even better firewals.
- Automatation of topic modeling and naratives extraction.
- Testing how LLM infected by such information will responde to basic questions.


**3. HOW?**  - TODO:
First of all we we are investigating two well-known Data Science methodologies for project management, namely, [CRISP-DM](https://ml-ops.org/content/crisp-ml) (Cross-Industry Standard Process for Data Mining) and [Microsoft TDSP](https://deeperinsights.com/ai-blog/how-to-run-a-data-science-team) (Team Data Science Process).

TODO: 
- building classifier;
- LDA and topic modeling;
- LangChain + opensource LLM;


## Literature review
The study by Pavlyshenko (2023) explores the efficacy of fine-tuning the Llama 2 large language model for detecting disinformation and fake news. The research demonstrates the model's advanced capabilities in text analysis, fact-checking, and sentiment extraction, making it a promising tool for enhancing automated fake news detection systems. The approach leverages PEFT/LoRA-based techniques to achieve substantial improvements in identifying complex narratives and disinformation.
But we are going to start from some baselines line in the article Kuzmin, G., Larionov, D., Pisarevskaya, D., & Smirnov, I. (2020). This paper by Gleb Kuzmin and colleagues, presented at the 3rd International Workshop on Rumours and Deception in Social Media, investigates fake news detection in Russian. The study compares various models using language features like bag-of-n-grams, Rhetorical Structure Theory features, and BERT embeddings. The research differentiates between satire and fake news, achieving high F1-scores in both binary and multiclass classifications, demonstrating the models' effectiveness in handling this challenging task.

Among additional studies, it is worth highligting the group of studies on the Systematic Mapping: These studies provide a broader overview of the research landscape in this field - some of the noteworthy mentions include
- "[Advanced Machine Learning techniques for fake news (online disinformation) detection: A systematic mapping study]([url](https://www.researchgate.net/publication/342394283_A_systematic_mapping_on_automatic_classification_of_fake_news_in_social_media))" by João Souza et al. details advanced machine learning techniques for fake news detection
- "[Online Fake News Detection Using Machine Learning Techniques: A Systematic Mapping Study]([url](https://www.researchgate.net/publication/357094499_Online_Fake_News_Detection_Using_Machine_Learning_Techniques_A_Systematic_Mapping_Study))" 2022 by Lahby et al, which details application of the SVM technique, and Deep Neural Network (DNN)

Studies on specific technicues explore the topic deeper into the application of particular machine learning algorithms for fake news detection: the relevant mentions include
- "[Detecting Fake News using Machine Learning: A Systematic Literature Review]([url](https://arxiv.org/list/cs.LG/recent))" (Discusses the use of Random Forest, Recurrent Neural Networks, and K-Nearest Neighbors) arXiv on detecting fake news using machine learning
- "[A Comparative Study of Machine Learning and Deep Learning Techniques for Fake News Detection]([url](https://www.mdpi.com/2078-2489/13/12/576))" 2022 by Alhamidi et al. (Analyzes Support Vector Machines and Deep Neural Networks) MDPI on machine learning vs deep learning for fake news detection



## Proposed datasets
1. [Fake and real news dataset on kaggle.com](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset.)
2. [ Propaganda Diary. Vox Ukraine](https://russiandisinfo.voxukraine.org/en/narratives)


## Methods : TODO

1. For the visualization task, we will use the library ```matplotlib, seaborn``` and ```plotly```. 
2. For the clustering part, we will cluster the areas according to the product consumption of their products using the two following methods:
  2.1. Apply the k means algorithm on the data (need also to find suitable k)
  2.2. Try to geographically validate the clustering using the following methods:
    * Use techniques seen in class to select the right value of ```k``` and  use dimensionality reduction to visualize the clusters in 2D
    * Find relevant geographical related metrics to evaluate the good-ness of fit. They will help us find formal evidence that there is (or not) a relation between geography and the clustering. For instance, we could use different distance metrics when the silhouette score is computed. We also plan to use a graph-based approach to compute such metrics (vertices are areas and edges link two areas if they share a physical border). 
3. Regarding the analysis of the clustering output, the following methods will be applied:
  3.1. For each cluster compute its typical product (average of the typical product of the areas contained in the cluster) and study the differences observed. We will then relate those differences in terms of the metabolic syndrome-related to diabetes prevalence (found in the Tesco paper).
  3.2. Quantify the predictive power of the clustering assignments to assess the information contained in the clustering. In order to do so, we will replicate the regression model (table 2. of the paper) on the number of diabetes prevalence in London and analyse the improvements (```R2``` for instance) when the clusters are added as dependent variable to the model compared to the base model of the paper. 
4. Conclude by proposing other possible practical usages of the clustering output.


## Project structure and installation
### Organisation of the repository
In order to be able to run our notebook, you should have a folder structure similar to:

    .
    ├── data                                      # Data folder
    │ ├── diabetes_estimates_osward_2016.csv      # [Diabetes estimates Osward](https://drive.google.com/drive/folders/19mY0rxtHkAXRuO3O4l__S2Ru2YgcJVIA)
    │ ├── all                                     # folder containing [Tesco grocery 1.0](https://figshare.com/articles/Area-level_grocery_purchases/7796666)
    │ │  ├── Apr_borough_grocery.csv              # example of file
    │ │  ├── ...
    │ ├── statistical-gis-boundaries-london       # folder containing the unzipped [Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london) 
    │ │  ├── ESRI                                 # contains the data to be loaded by geopandas
    │ │  │  ├── London_Borough_Excluding_MHW.dbf  # example of file
    │ │  │  ├── ...
    │ │  ├── ...
    ├── images                              # Contains the ouput images and html used for the data story
    ├── extension.ipynb                     # Deliverable notebook for our extension
    ├── vizu.ipynb                          # Notebook containing only the vizualisations (if the reader only was to see the interactive viz)
    ├── Data Extraction.ipynb               # Notebook that generates the subset of tesco used in this analysis
    └── README.md               
    

### How to run the code
Follow the steps in Data Acquisition to download the raw datasets
Follow the steps in Data Processing to generate the preprocessed data
Data exploration: run the data_exploration/data_exploration.ipynb notebook to see the data exploration steps taken.
Embeddings: open the models/embedding folder:
Autoencoder: follow the steps in Train Autoencoders to understand how to train and use the available autoencoders
PCA: run the pca_embedding.ipynb notebook to create the PCA embedding.
Rule-based: follow the steps in Build Rule-based features to generate preprocessed data usefull for performance comparision.
Clustering: run the models/clustering/Kmeans.ipynb notebook to see the code related to the clustering.
Profitablity prediction: run the models/prediction/prediction.ipynb notebook for the profitablity prediction task.

### Dependencies requirement

Furthermore, you should have the following additional libraries installed. In the repository, we provide a `requirement.txt` file from which you can create a virtual python environment.
| Library                         | Versino                    |
|:--------------------------------| :--------------------------|
| pandas                          |...                         |
| numpy                           |...                         |


## References:
Pavlyshenko, B. (2023). Analysis of Disinformation and Fake News Detection Using Fine-Tuned Large Language Mode. ArXiv. [https://doi.org/10.31234/osf.io/76xfs](https://arxiv.org/pdf/2309.04704)

Kuzmin, G., Larionov, D., Pisarevskaya, D., & Smirnov, I. (2020). Fake news detection for the Russian language. In A. Aker & A. Zubiaga (Eds.), Proceedings of the 3rd International Workshop on Rumours and Deception in Social Media (RDSM) (pp. 45–57). Association for Computational Linguistics. Retrieved from https://aclanthology.org/2020.rdsm-1.5



## Timeline and contributions:

### Week 1 : Data Acquisition & Project Setup

| Task                                | Team member(s)                  | work hours  |
| :-----------------------------------|:--------------------------------| -----------:|
| Idea research and concepts          |                                 |             |
| Data search                         |                                 |             |
| Git utilisation and usage           | All team members                | 3h          |
| Environment setup                   |                                 | 2h          |
| Data fetching script test           |                                 | 3h          |
| Data fetching validation            | Vadym                           | 2h          |
| Data fetching improvements          | Vadym                           | 2h          |


### Week 2 : Data preprocessing and literature review

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Data cleaning                            | Vadym                           | 5h          |
| Data exploration paper dataset           | Vadym, Petro                    | 2h          |
| Literature review                        | Oleksandra, Petro               | 3h          |
| Raw data => embedding format             | Vadym                           | 3h          |


### Week 3 : Embedding & Clustering 

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Autencoder       basic code              | Vadym                           | 3h          |
| Comparision with PCA and debugging       | Vadym                           | 1h          |
| K-means                                  | Vadym, Petro                    | 2h          |

### Week 4 : Clustering analysis, Profitablity prediction & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Clustering analysis                     | Zoryana, Vadym                  | 4h          |
| Github pages setup                      | Vadym                           | 2h          |
| Data story (1)                          | Petro, Oleksandra               | 5h          |
| Data story (2)                          | Bohdan                          | 2h          |

### Week 5 : Improvements in data processing & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Token based scaling                     | Zoryana, Vadym                  |    5h        |
| Token one hot encoding                  | Zoryana, Vadym                  |    1h        |
| Deep NN                                 | Vadym                           |    1h        |
| Better data processing                  | Bohdan                          |    2h        |
| Improved data exploration               | Bohdan                          |    3h        |
| Better understanding of PCA output      | Bohdan                          |    1h        |
| Data story (3)                          | Lucas                           |    1h        |
| Performance comparision                 | Vadym                           |    2h        |        


## Total contribution:

| Team member                     | work hours   |
|:--------------------------------| ------------:|
| Vadym                           |    81h       |
| Zoryana                         |    60h       |
| Petro                           |    60h       |
| Bohdan                          |    60h       |
| Oleksandra                      |    60h       |
