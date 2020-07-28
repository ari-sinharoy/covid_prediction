'''
Exploratory analysis of the covid data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
f_path = r'C:\\Users\\JJ\\Desktop\\Data Science Projects\\India_Covid-19_Prediction'
os.chdir(f_path)
dat_f = pd.read_csv('data_full.csv')

def dat_con(dat, con):
    return dat[dat.location == con]

'''
Our hypothesis is that new_cases and stringency_index should be strongly
correlated. If not, implementation of the lockdown must have been poor for
various reasons. We calculate pearson correlation ratio and probability for
all the countries in order to justify the hypothesis 
'''
pearson_coef = pd.DataFrame(columns = ['location','pearson_r','probability'])
for country in np.unique(dat_f.location):
    test_data = dat_con(dat_f, country)
    r,p = stats.pearsonr(test_data.stringency_index, test_data.new_cases)
    pearson_coef = pearson_coef.append({'location': country,
                                        'pearson_r': r,
                                        'probability': p}, ignore_index = True)
    
corr_cases = list(pearson_coef[((abs(pearson_coef.pearson_r) > 0.5) & 
              (pearson_coef.probability < 1.0e-05))].location)  
    
def plt_corr(dat, con):
    dx = dat_con(dat, con)
    y1 = dx.new_cases/dx.new_cases.max()
    y2 = dx.stringency_index/100
    y3 = dx.total_cases/dx.total_tests
    plt.title("Correlation Between Stringency & New Cases: "+con)
    plt.plot(range(len(y1)), y1, 'b', 
             range(len(y2)), y2, 'r', 
             range(len(y3)), y3, 'g')

static_var = ['location','total_cases', 'population', 'pop_density', 'aged_65', 
              'gdp_per_capita', 'life_exp_data', 'smoking_prev', 
              'tb_incidence', 'int_arrival', 'literacy_rate', 
              'stringency_index']

static_dat = pd.DataFrame()
country_list = np.unique(dat_f.location)
for item in country_list:
    static_dat = static_dat.append(pd.DataFrame(dat_con(
            dat_f,item)[static_var].iloc[-1,:].values[np.newaxis]))
static_dat.columns = static_var    
static_dat.reset_index(drop = True, inplace = True)

# Calculate & plot correlation matrix
corr_df = abs(static_dat.iloc[:,1:].astype(float).corr())
fig = plt.figure(figsize=(12,9))    
hmp = sns.heatmap(corr_df, cmap='coolwarm', annot=True, 
                  fmt='.2f',annot_kws={'size':12})
hmp.set_xticklabels(hmp.get_xticklabels(), rotation=35, ha='right')
plt.show()

'''
Total cases are strongly correlated with population, international arrival,
stringency index, and gdp per capita. We ignore life expectancy because it is 
strongly correlated with gdp per capita.
'''

# Countries where covid crisis is over / fully under control
cvd_over = pd.DataFrame(columns = ['location','total_cases','population'])
for item in country_list:
    dx = dat_con(dat_f,item)
    dxn = dx.iloc[-1]
    if (dxn.new_cases / dx.max().new_cases) <= 0.02:
        cvd_over = cvd_over.append({'location': item,
                                    'total_cases': dxn.total_cases,
                                    'population': dxn.population},
                                    ignore_index = True)
cvd_over = cvd_over.sort_values(by = ['total_cases'], 
                                ascending = False)

'''
Only Italy and Germany had more than 10^5 total cases among the countries
which fully controlled the crisis by our definition (latest new cases less 
than 5% of the maximum occurance of new cases in a day. We will use these 
two countries as our model. 
'''

'''
It is interesting to note than Malaysia, Thailand, Sri Lanka, Mali, Niger,
and Vietnam have surprisingly low cases inspite of sizable populations. Hence
we further probe the data from those countries
'''

con_abb = ['Malaysia', 'Thailand', 'Sri Lanka', 'Mali', 'Niger', 'Vietnam']
con_abb_df = static_dat[static_dat.location.isin(con_abb)]
abb_mean = con_abb_df.iloc[:,1:].astype('float').mean()

# Let's compare with other asian countries
con_asia = ['China', 'India', 'Indonesia', 'Pakistan', 'Bangladesh', 'Russia',
            'Japan', 'Philippines', 'Iran', 'Turkey', 'South Korea']
con_asia_df = static_dat[static_dat.location.isin(con_asia)]
rest_mean = con_asia_df.iloc[:,1:].astype('float').mean()

comb_mean = pd.concat([abb_mean,rest_mean], axis = 1, sort = False)
comb_mean.columns = ['select_countries', 'rest_of_asia']

'''
We find that population density, 65+ aged population, smoking prevelence, 
tb incidence, and internationa arrival in the selected countries are 
significantly lower compared to the rest of the asian countries selected  
'''

def tab_x(start, end, dim):
    step = (end - start)/dim
    return [start + i*step for i in range(dim)]