'''
Pull covid-19 data from ourworldindata.org
'''

import pandas as pd
import numpy as np

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
dat_2 = dat_2[dat_2.location != 'Hong Kong']

# Group the data by location
grouped_dat = dat_2.groupby(['location'])

class con_dat:
    def __init__(self, n = ""):
        self.country = n
    
    def get_value(self):
        dx = grouped_dat.get_group(self.country)
        dx = dx[dx.new_cases > 0]
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
        elif dx.total_tests.sum() < dx.new_tests.sum():
            dy = dx.new_tests
            ser = [dy.iloc[:i+1].sum() for i in range(len(dy))]
            dx.loc[:,'total_tests'] = ser
        return dx
    
country_list = list(np.unique(dat_2.location))

# storing the data as csv
import csv
filename = 'covid_data.csv'

with open(filename, 'w', newline = '') as csvfile:
    
    csvwriter = csv.writer(csvfile)
    
    # add header
    csvwriter.writerow(new_cols)
    
    # create and add the rows
    for item in country_list:
        x = con_dat(item)
        y = x.get_value()
        z = y.values.tolist()
        csvwriter.writerows(z)