'''
Pull covid-19 data from ourworldindata.org
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'

# Read csv from the site directly
dat_1 = pd.read_csv(url,error_bad_lines=False);
dat_1.head()
col_names = list(dat_1.columns.values)

dat_1['date'] = pd.to_datetime(dat_1['date'],format='%Y-%m-%d')

# Keep relevant features only
new_cols = ['location','date','new_cases','new_deaths','new_tests',
            'total_cases','total_deaths','total_tests','stringency_index',
            'population','population_density','aged_65_older','gdp_per_capita',
            'female_smokers','male_smokers','life_expectancy',
            'hospital_beds_per_thousand']

dat_2 = dat_1[new_cols]

# Group the data by location
grouped_dat = dat_2.groupby(['location'])

class con_dat:
    def __init__(self, n = ""):
        self.country = n
    
    def get_value(self):
        dx = grouped_dat.get_group(self.country)
        dx = dx[dx.new_cases > 10]
        for i in range(2,8):
            if np.isnan(dx.iloc[0,i]):
                dx.iloc[0,i] = 0.0
        dx = dx.interpolate().fillna(
               method = 'ffill')
        # filling in new_tests values from total_tests values
        if dx.new_tests.sum() == 0:
            dy = dx.total_tests
            ser = [0] + [dy.iloc[i+1]-dy.iloc[i] for i in range(len(dy)-1)]
            dx.loc[:,'new_tests'] = ser
        return dx



# check for missing data for india
dat_ind = dat_1.loc[dat_1.location == "India"]
dat_ind.isna().sum()/len(dat_ind)

dat_ind = dat_ind.fillna(method = 'ffill')
dat_ind = dat_ind.fillna(method = 'bfill')

# check for missing data for india
dat_ctr = dat_1.loc[dat_1.location == "Pakistan"]
dat_ctr.isna().sum()/len(dat_ctr)

dat_ctr = dat_ctr.fillna(method = 'ffill')
dat_ctr = dat_ctr.fillna(method = 'bfill')