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

<img src="image.png"></img>


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
