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
