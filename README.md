# Covid-19 Predictive Analysis

### Hypothesis 
In absence of any treatment and/or a vaccine, covid-19 curves of all the countries are going to be extremely similar. The government intervention and the population / population density are likely to have an effect on the exact shape of the curve, the uptake and the decay rates. We assume that those factors can be introduced as mulplicative factors in the model system. In this analysis I use the data of **_Italy_** as the model. Among the countries where the crisis (or the first wave) got over, the total cases in Italy was significantly higher than the rest of the countries in that cohort and that is the reason behind using Italy as the model country.

### Data Source
1. Covid time-series data is pulled from [ourworldindata.org](https://covid.ourworldindata.org/data/owid-covid-data.csv) 
   - the data for the current analysis is pulled on 22 July, 2020
   - the python code is saved in data_pull.py
   - the cleaned data is stored in covid_data.csv
2. Additional demographic data and some missing data are pulled from the world bank database
   - the python code is saved in data_consolidation.py
   - added the following features:
     - aged 65+ population 
     - gdp per capita, 
     - life expectancy data, 
     - population density 
     - smoking prevalence 
     - tuberculosis incidence 
     - international arrival
   - the cleaned data is stored in data_full.csv
3. Exploratory analysis is done in exploratory_analysis.py
4. The predictive analysis is done in cvd_prediction.py
   - use least square method to fit the Italy-data by a bimodal skewed gaussian distribution 
   - measure the pearson correlation between the data from any country and the scaled model data, sliced at different positions
   - pick the slice with the highest pearson r to represent the covid curve in the investigating country
   - predict the total cases, maximum daily cases, the mid-point and the end-point
   - the summary is stored in pred_summary.csv
   - the predicted time series data for all the countries are stored in pred_timeseries.csv
