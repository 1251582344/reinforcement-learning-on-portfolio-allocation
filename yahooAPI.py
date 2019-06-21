# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:25:40 2019

@author: 佟斌
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import pandas_datareader as web
 

import datetime


sp_name=list(pd.read_csv("sp500.csv")['Ticker'])
"change the start and end time below to customize any period"
start = datetime.datetime(2006, 9, 1) 
end = datetime.datetime(2018, 12, 31)

"use google's period as the benchmark"
df_std = list(web.DataReader('GOOG', 'yahoo', start, end).index)

result=pd.DataFrame()
result['date']=df_std

for i in sp_name:   
    try:
        tp_df=web.DataReader(i, 'yahoo', start, end)
        df2 = list(tp_df.index)
        if df2==df_std:
            result[i+"_price"]=list(tp_df['Close'])
            print(i,"  ",result.shape[1],np.mean(result[i+"_price"]))
           
    except:
        continue
    
result.to_csv("SP500_data.csv")

            
