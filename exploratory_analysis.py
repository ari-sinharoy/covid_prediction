'''
Exploratory analysis of the covid data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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
    plt.title("Correlation Between Stringency & New Cases: "+con)
    plt.plot(range(len(y1)), y1, 'b', 
             range(len(y2)), y2, 'r')

static_var = ['total_cases', 'population', 'pop_density', 'aged_65', 
              'gdp_per_capita', 'life_exp_data', 'smoking_prev', 
              'tb_incidence', 'int_arrival', 'literacy_rate', 
              'stringency_index']

static_dat = pd.DataFrame()
country_list = np.unique(dat_f.location)
for item in country_list:
    static_dat = static_dat.append(pd.DataFrame(dat_con(
            dat_f,item)[static_var].iloc[-1,:].values[np.newaxis]))
static_dat.columns = static_var    

# Calculate & plot correlation matrix
corr_df = abs(static_dat.corr())

fig = plt.figure(figsize=(12,9))    
hmp = sns.heatmap(corr_df, cmap='coolwarm', annot=True, 
                  fmt='.2f',annot_kws={'size':12})
hmp.set_xticklabels(hmp.get_xticklabels(), rotation=35, ha='right')
plt.show()

'''
Total cases are strongly correlated with population, international arrival,
stringency index, gdp per capita. We ignore life expectancy because it is 
strongly correlated with gdp per capita.
'''