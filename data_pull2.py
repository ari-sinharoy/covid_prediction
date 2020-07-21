'''
Collect and merge additional data
'''

import pandas as pd
import numpy as np

df = pd.read_csv("covid_data.csv")

'''
Stringency index is an extremely important parameter in measuring the covid-19
control, we will drop the countries for which this parameter value is missing.
'''

del_con = []
for item in np.unique(df.location.values):
    if df[df.location == item]['stringency_index'].isna().sum() > 0:
        del_con += [item]

for item in del_con:        
    df = df[df.location != item]
    
'''
The rest of the missing data for pop_density, aged_65+, gdp_per_capita,
smoking_pop, life_expectancy, hosp_capacity can be obtained from world bank
database.
We will also add xx more features, namely:
    1. urban_pop
    2. literacy_rate (impact awareness / ability to follow instructions)
    3. international_arrival (influence initial phase of transmission)
    4. tb_incidence
'''

literacy_data = pd.read_csv('literacy_data.csv')
literacy_data.columns = ['location', 'code', 'year', 'literacy_rate']
literacy_data = literacy_data.groupby(['location']).mean().reset_index()
literacy_data = literacy_data[['location','literacy_rate']]

dat_lst = ["aged_65", 'gdp_per_capita', 'life_exp_data', 'pop_density', 
           'smoking_prev', 'tb_incidence', 'int_arrival']


df_wbank = pd.DataFrame({'Country Name': list(np.unique(df.location))})
for item in dat_lst:
    pdx = pd.read_csv(item+'.csv', skiprows = 3)
    pdx[item] = pdx.iloc[:,-5:].mean(axis = 1)
    pdx = pdx[['Country Name',item]]
    df_wbank = pd.merge(df_wbank,pdx)

df_wbank = df_wbank.rename(columns = {'Country Name': 'location'})
df_wbank = pd.merge(df_wbank, literacy_data)

# Replace the missing values with the column mean
df_wbank = df_wbank.fillna(df_wbank.mean())

df2 = df[['location','date','new_cases','new_deaths','new_tests',
         'total_cases','total_deaths','total_tests',
         'stringency_index','population']]

dff = pd.merge(df2,df_wbank)

dff.to_csv(r"C:\Users\JJ\Desktop\Data Science Projects\India_Covid-19_Prediction\data_full.csv",
           header=True)