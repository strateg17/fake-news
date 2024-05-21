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
TODO:



## Proposed datasets
1. [Fake and real news dataset on kaggle.com](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset.)
2. [ Propaganda Diary. Vox Ukraine](https://russiandisinfo.voxukraine.org/en/narratives)
3. [Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london) -- This dataset will help us to relate the areas of the Tesco dataset with their real geographical positions, needed for the visualization.



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


## Timeline and contributions:

### Week 1 : Data Acquisition & Project Setup

| Task                                | Team member(s)                  | work hours  |
| :-----------------------------------|:--------------------------------| -----------:|
| Idea research and concepts          |                                 |             |
| Data search                         |                                 |             |
| Git utilisation and usage           | All team members                | 3h          |
| API choice and query design         |                                 | 4h          |
| Environment setup                   |                                 | 2h          |
| Data fetching script test           |                                 | 3h          |
| Data fetching validation            | Augustin                        | 2h          |
| Data fetching improvements          | Augustin                        | 2h          |


### Week 2 : Data preprocessing

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Data cleaning                            | Augustin                        | 5h          |
| Data exploration paper dataset           | Augustin                        | 2h          |
| Data exploration                         | Lucas                           | 3h          |
| Raw data => embedding format             | Lucas                           | 3h          |


### Week 3 : Embedding & Clustering 

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Autencoder keras basic code              | Lucas                           | 3h          |
| Comparision with PCA and debugging       | Lucas                           | 1h          |
| K-means                                  | Augustin                        | 2h          |

### Week 4 : Clustering analysis, Profitablity prediction & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Clustering analysis                     | Lucas                           | 4h          |
| Profitablity prediction setup           | Augustin                        | 2h          |
| Github pages setup                      | Lucas                           | 2h          |
| Data story (1)                          | Lucas                           | 5h          |
| Data story (2)                          | Augustin                        | 2h          |

### Week 5 : Improvements in data processing & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Token based scaling                     | Lucas & Augustin                |    5h        |
| Token one hot encoding                  | Lucas                           |    1h        |
| Token encoding in profitablity prediction   | Augustin                    |    1h        |
| Deep NN for   profitablity prediction   | Augustin                        |    1h        |
| Better data processing                  | Augustin                        |    2h        |
| Improved data exploration               | Lucas                           |    3h        |
| Better understanding of PCA output      | Augustin                        |    1h        |
| Autencoder testing                      | Augustin                        |    2h        |
| Data story (3)                          | Lucas                           |    1h        |
| Add ruled based indicators for autoencoder performance comparision | Lucas |    2h       |        

### Week 6 : Hyperparameter opmisation, improvements & report writing

| Task                                    | Team member(s)                  | work hours   |
| :---------------------------------------|:--------------------------------| ------------:|
| Filter illiquid data & debug            | Lucas                           |    3h        |
| Update architecture for liquid data     | Augustin                        |    3h        |
| Research on attention learning          | Lucas                           |    2h        |
| Data processing simpler pipeline        | Augustin                        |    2h        |
| Autencoder improvement and debug        | Augustin                        |    3h        |
| Autencoder manual tests for several architectures        | Augustin       |    8h        |
| Testing optimizers                      | Augustin                        |    2h        |
| Talos setup                             | Lucas                           |    2h        |
| Running Talos                           | Augustin                        |    1h        |
| Hyperparameter opmisation               | Lucas & Augustin                |    4h        |
| Reporting losses                        | Augustin                        |    1h        |
| Kmeans : better silouhette analysis     | Lucas                           |    3h        |
| Kmeans : update results for liquid data | Lucas                           |    4h        |
| PCA embedding                           | Lucas                           |    1h        |
| Ruled based data : pandas-ta implementation    | Lucas                    |    1h        |
| Ruled based data : pandas implementation       | Lucas                    |    3h        |
| Ruled based data : code optimisation           | Lucas                    |    3h        |
| Ruled based data : performance comparision with AE  | Lucas               |    1h        |
| Repository cleaning                     | Lucas & Augustin                |    1h        |
| Notebook comments and markdown          | Lucas & Augustin                |    4h        |
| Data story (4)                          | Lucas & Augustin                |    6h        |

## Total contribution:

| Team member                     | work hours   |
|:--------------------------------| ------------:|
| Lucas Giordano                  |    81h       |
| Augustin Kapps                  |    60h       |
