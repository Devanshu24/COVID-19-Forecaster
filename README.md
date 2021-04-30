# COVID-19 Forecaster

**Disclaimer: We use machine learning methods to forecast the spread of the virus, hence it is suscpetible to uncertainty and potentially errors.**

## Usage

```console
$ python src/main.py --help
usage: main.py [-h] {lstm,arima,fbprophet,nbeats,sir,glm,persistence}

Analyse various algorithms

positional arguments:
  {lstm,arima,fbprophet,nbeats,sir,glm,persistence}
                        Selection of model

optional arguments:
  -h, --help            show this help message and exit

$ python src/main.py persistence

```
*Note:* The results in the paper are reported after doing cross validation, hence running a single instance of the problem may not match the said results.
## Data

```console
data
├── raw (Raw Data from various sources)
│   ├── 2020_IN_Region_Mobility_Report.csv (https://www.google.com/covid19/mobility/)
│   ├── 2021_IN_Region_Mobility_Report.csv (https://www.google.com/covid19/mobility/)
│   ├── applemobilitytrends-2021-04-29.csv (https://covid19.apple.com/mobility)
│   ├── Global_Covid_OWID_(30-01-2020->19-04-2021).csv (https://github.com/owid/covid-19-data/tree/master/public/data)
│   ├── NIFTY_50(01-02-2020->19-04-2020).csv (https://www1.nseindia.com/)
│   ├── Region_Mobility_Report_CSVs.zip (https://www.google.com/covid19/mobility/)
│   ├── S&P_BSE_SENSEX(01-02-2020->19-04-2020).csv (https://www.bseindia.com/)
│   ├── state_wise_daily.csv (https://api.covid19india.org/)
│   └── statewise_tested_numbers_data.csv (https://api.covid19india.org/)
├── district_mobility_cases.csv (Data of Mobility statistics on a district level)
├── India_OWID.csv (Cleaned data from Our World in Data)
├── India_OWID_reduced.csv (Cleaned data from Our World in Data (reduced features))
├── India_OWID_with_mobility_data.csv (Data of Mobility statistics on a country level)
├── IN_Region_Mobility_Report.csv (Mobility data for India)
├── regional_mobility.csv (Mobility data for India on a regional level)
├── state_mobility_cases_agg=max_2020.csv (Aggregated Mobility statistics for mobility derived from district data)
├── state_mobility_cases_agg=max_2021.csv (Aggregated Mobility statistics for mobility derived from district data)
├── state_wise_cases.csv (State wise case counts)
└── UK_data.csv (Case Data for UK)

1 directory, 19 files
```
*Note:* For interactive visualizations please refer to `notebooks/Visualization.ipynb`
