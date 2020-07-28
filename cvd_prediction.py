'''
Our goal is to predict the timeline for completion of the crisis in selected
countries and provide validatory argument based on the data analysis.
'''

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import math
#from scipy.optimize import curve_fit
import scipy.special as sp
from scipy.optimize import least_squares

'''
We use the Italy and Germany data as models in predicting the status of the
covid-19 crisis in other countries
'''
import os
f_path = r'C:\\Users\\JJ\\Desktop\\Data Science Projects\\India_Covid-19_Prediction'
os.chdir(f_path)
dat_f = pd.read_csv('data_full.csv')

def dat_con(dat, con):
    return dat[dat.location == con]

def max_sc(dat):
    return dat/dat.max()

# Define moving average with padding
def mov_avg(ser,n):
    dim = len(ser);
    app = pd.Series([ser.iloc[0]]*n);
    ext = pd.Series([ser.iloc[-1]]*n);
    ser_pad = app.append(ser,ignore_index=True);
    ser_pad = ser_pad.append(ext,ignore_index=True);
    return pd.Series([ser_pad.iloc[j-n:j+n+1].mean() for j in range(n,dim+n)])

# We select Italy as the model country for the primary analysis

def skewnorm(x, x_s, x_m, a, b, x_s2, x_m2,b2):
    ndf = (1/(x_s*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-x_m)/x_s,2)))
    err = 0.5*(1+sp.erf(a*((x-x_m)/x_s)/(np.sqrt(2))))
    ndf2 = (1/(x_s2*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-x_m2)/x_s2,2)))
    return 2*b*ndf*err + b2*ndf2

def f_res(p,x,y):
    return y-skewnorm(x,*p)

# Model data #1. Italy
y1 = dat_f[dat_f.location == 'Italy'].new_cases
y1_train = mov_avg(y1,5)
y1m = y1_train.max()
y1_train /= y1m
x1_train = np.linspace(0,1,len(y1_train))    
p0 = np.ones(7);
res1 = least_squares(f_res,p0,args=(x1_train,y1_train))    

con_all = np.unique(dat_f.location)

def pred_1(dat, x2_min):
    mw = 1
    u1 = mov_avg(dat.new_cases, mw);
    u1_mx = u1.max()
    dim = len(u1)
    u1 /= u1_mx
    lm2 = int(50)
    if y1m > u1_mx:
        x2_v = np.linspace(1.0+x2_min, 2.0, lm2)
    else:
        x2_v = np.linspace(x2_min, 1.0, lm2)
    diff = pd.DataFrame(columns = ['m1','m2','cost1','cost2'])
    for x1 in np.linspace(0.1, 1.0, 100):
        for x2 in x2_v:
            dx1 = skewnorm(np.linspace(0,x1,dim),*res1.x)
            #dx2 = skewnorm(np.linspace(0,x1,dim),*res2.x)        
            dx3 = x2*u1
            diff = diff.append({'m1': x1,
                                'm2': x2,
                                'cost1': stats.pearsonr(dx1, dx3)},
                                ignore_index = True)

    diff1 = diff.sort_values(by = ['cost1'], ascending = False)
    
    m1_op = diff1.iloc[0].m1
    m2_op = diff1.iloc[0].m2
    
    ser1 = mov_avg(y1_train*(u1_mx/m2_op), mw)
    d1 = np.linspace(0,dim/m1_op,len(ser1))
    
    dfx1 = pd.DataFrame(columns = ['x','y'])
    dfx1['x'] = d1
    dfx1['y'] = ser1
   
    return dfx1.reset_index(drop=True)


def trial(dat):
    dfx = pred_1(dat, 0.3)
    tot_cs = sum([(dfx.iloc[i+1].x - dfx.iloc[i].x)*dfx.iloc[i+1].y 
                  for i in range(len(dfx)-1)]) + dfx.iloc[0].y
    max_nc = dfx.y.max()
    mid_p = dfx.loc[np.where(dfx.y == max_nc)[0][0]].x
    end_p = dfx.iloc[-1].x*0.8
    
    return [int(tot_cs), int(max_nc), int(mid_p), int(end_p), dfx]
    

def pred_summary(dat, con_lst):
    dff = pd.DataFrame(columns = ['location', 'days', 'new_cases'])
    
    dfy = pd.DataFrame(columns = ['location',
                                  'max_total_cases',
                                  'max_new_cases',
                                  'pred_mid_date',
                                  'pred_end_date'])
    for con in con_lst:
        dat0 = dat_con(dat, con)
        dat1 = trial(dat0)
        dfy = dfy.append({'location': con,
                          'max_total_cases': dat1[0],
                          'max_new_cases': dat1[1],
                          'pred_mid_date': dat1[2],
                          'pred_end_date': dat1[3],}, ignore_index = True)
        dfi = dat1[-1]
        nm = len(dfi)
        dfi['location'] = [con]*nm
        dfi = dfi[['location', 'x', 'y']]
        dfi[['x', 'y']] = dfi[['x', 'y']].astype(int)
        dfi.columns = ['location', 'days', 'new_cases']
        dff = dff.append(dfi, ignore_index = True)

    return dff, dfy

# Analyze countries with 10,000 or more total cases
con_test = [item for item in con_all 
            if dat_f[dat_f.location == item].total_cases.iloc[-1] >= 10000]

result_1 = pred_summary(dat_f, con_test)

pred_times = result_1[0]
pred_static = result_1[1]

def plt_test(con):
    dat_1 = dat_con(dat_f, con)
    y1 = dat_1.new_cases
    dat_2 = dat_con(pred_times, con) 
    y2 = dat_2.new_cases
    plt.plot(range(len(y1)), y1, 'r', linewidth = 4, label = 'Actual')
    plt.plot(mov_avg(dat_2.days,1), mov_avg(y2,1), 'b', linewidth = 3,
             label = 'Predicted')
    plt.title(con)
    plt.xlabel('Days')
    plt.ylabel('New Cases')
    plt.legend()

pred_times.to_csv('pred_timeseries.csv', index = False, header=True)
pred_static.to_csv('pred_summary.csv', index = False, header=True)

