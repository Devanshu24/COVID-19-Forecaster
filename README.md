# ML Major Project

## Logs

### 18th April 2021
#### Sharad
- Basic Model Exploration. End to end pipeline for the simplest model
- Basic Model considers:
  - Usual features
  - Mobility Data (Overall; Separated based on errand type)
- Basic Model fails:
  - LSTM
  - Basic Model suffers from distribution shift. In particular, it is not accounting for features in the future period. 
  - The final model just seems to predict somewhat of an average of the features as shown below

![image](https://user-images.githubusercontent.com/56106207/115334589-4bf06700-a1b9-11eb-843e-255d6cd07cd3.png)



- Thoughts
  - It seems critical that we need a model that is in some sense "evolving"
  - For region based modelling, perform ablation into individual features. And also, into region based plotting 

### 19th April 2021
#### Vybhav
- Transfer Learning
  - Tested the basic model (lagged mobility + testing data) on UK data as they've already had their second peak
  - No real improvement seen in results

![image](https://user-images.githubusercontent.com/81354041/115293507-0a3dcd00-a175-11eb-81d7-0169363d28a3.png)
![image](https://user-images.githubusercontent.com/81354041/115293539-145fcb80-a175-11eb-8c33-a5726174732a.png)

- Basic Model : Linear Regression
  - Features considered : 7 day lagged mobility, daily tests, daily deaths
  - Seems to do better than LSTM, though still pretty poor

![image](https://user-images.githubusercontent.com/81354041/115298034-8b4b9300-a17a-11eb-9809-bfb8f779eae6.png)
![image](https://user-images.githubusercontent.com/81354041/115298153-b9c96e00-a17a-11eb-9fcc-0114bc2ca6c0.png)

#### Devanshu
- Regional Mobility Analysis
  -  Most of the regional mobility trends are really great indicators of the future cases counts, a few examples for reference
  -  These are just the Grocery and Pharmacy change, and for a few states but it gives a flavour
  - ![maharashtra](https://user-images.githubusercontent.com/56106207/115298679-64da2780-a17b-11eb-8582-7d0196fab3b4.png)
  - ![kerala](https://user-images.githubusercontent.com/56106207/115298692-699edb80-a17b-11eb-8b9a-2f728a4eb0d8.png)
  - ![mp](https://user-images.githubusercontent.com/56106207/115298718-70c5e980-a17b-11eb-9275-0f2560532c1b.png)


#### Sharad
- Created Regional Mobility Aggregator Notebooks and DataFrames

### 20th April
#### Vybhav

- Basic Model : LSTM with 7 day lagged mobility
  - Model Summary
    - Used 7 day lagged mobility instead of direct mobility for each of the 6 categories
    - 7 day lagged mobility for category x on day **day** = Average mobility for category x from **day-7** to **day-2** (Reference : https://www.medrxiv.org/content/10.1101/2020.12.21.20248523v1.full )
    - Other features : cases, deaths, tests, tests per case and positive rate
    - Model parameters : n_features=11, n_hidden=512, seq_len=1, n_layers=2
    - Splitting done with TimeSeriesSplit(n_splits = 5)
    - Testing done with train_model_with_crossval() from the CovidPredictor notebook, with num_epochs set to 150

  - Results
    - R2 scores (5 splits) : (-24.726, 0.261, 0.202, 0.826, 0.643)
    - Very poor predictions for the initial stage of the pandemic, predicts a much faster rise in cases than was actually seen (First image)
    - Predicts the drop in cases after the first peak and the occurrence of the second peak relatively well

  ![image](https://user-images.githubusercontent.com/81354041/115383217-2f712080-a1f3-11eb-8d00-4f77e063848c.png) 
  ![image](https://user-images.githubusercontent.com/81354041/115362443-2a09db00-a1df-11eb-8925-fc2ffa305a42.png)

  - Questions/Thoughts
    - Setting window size (seq_len) to 1 improves performance drastically, unsure why?
 

