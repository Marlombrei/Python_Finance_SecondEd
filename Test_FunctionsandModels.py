from __future__ import division
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats as st
import pandas as pd
import pandas_datareader.data as web
import pandas.tseries
# from datetime import datetime, date, time
# from dateutil.parser import parse
# import csv
# import json
# import openpyxl
# from openpyxl import load_workbook
import matplotlib.pyplot as plt
#import matplotlib.finance as mfin



pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.5f}'.format
#pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 100
np.set_printoptions(linewidth=500)

"""
#===============================================================================
# Chapter 01 
#===============================================================================
"""
# import chapter_01 as chp1
# 
# cf = [-100,-30,10,40,50,45,20]
# 
# print(chp1.simple_npv(rate=0.035, cash_flows=cf))
# 
# 
# print(chp1.simple_npv_enumerate(rate=0.035, cash_flows=cf))
# 
# 
# cashFlows=(550,-500,-500,-500,1000)
# r = 0
# 
# # while r<1:
# #     r += 0.000001
# #     npv = chp1.simple_npv(r,cashFlows)
# #     if  abs(npv <= 0.0001):
# #         print(r)
#     
# url=url='http://canisius.edu/~yany/data/ibm.csv'
# x = pd.read_csv(url)
# print(x)

"""
#===============================================================================
# Chapter 03
#===============================================================================
"""
# print(sp.fv(rate=0.1, nper=2,pmt=0,pv=-100))
# 
# print(sp.pv(rate=0.0145, nper=5, pmt=0, fv=234, when='end'))


"""
#===============================================================================
# Chapter 04
#===============================================================================
"""
# for i in dir(web):
#     print(i)

# df = web.get_data_google('NYSE: IBM')

# vix = web.DataReader("VIXCLS", "fred")
# print(vix.tail())
# 
# ff =web.DataReader("F-F_Research_Data_Factors", "famafrench")
# print(ff[1])


"""
#===============================================================================
# Chapter 05
#===============================================================================
"""
# import Bond_and_Stock_Valuation as bsv
# 
# # print(bsv.Rc2Rm(0.02, 2))
# 
# # Interest Rates
# """ Technically speaking, YTM is the
# same as Internal Rate of Return (IRR)"""
# 
# times = [3/12, 6/12, 2,3,5,10,30]
# rates = [0.47, 0.6, 1.18, 1.53, 2, 2.53, 3.12]
# 
# # plt.plot(times, rates)
# # plt.show()
# 
# print(bsv.pandas_interpolate(times, rates))
# 
# 
# 
# print(bsv.duration(t=[0,1,2],
#                    cash_flow=[-100,100,100],
#                    y=0.05))
# 
# print(bsv.YTM(5, 0.03*1000, -825, 1000))
# 
# 
# p0 = sp.pv(0.04, 15, 0, -100)
# print(p0)
# 
# 
# p1 = sp.pv(0.05, 15, 0, -100)
# print(p1)
# 
# print((p1 - p0)/p0)
# 
# print('\n\n')
# 
# b1 = sp.pv(0.04, 30, -9, -100)
# print(b1)
# 
# b2 = sp.pv(0.05, 30, -9, -100)
# print(b2)
# 
# print((b2-b1)/b1)
# 
# 
# print(bsv.durationBond(0.1, 0.04, 4))
# 
# 
# t = 30
# face_value = 1000
# coupon_rate = 0.035
# total_face_value = 60000000


"""
#===============================================================================
# Chapter 06
#===============================================================================
"""
# import chapter_06 as chp6
# import statsmodels.api as sm
# 
# y = [1,2,3,4,2,3,4]
# x = range(1,8)
# 
# chp6.OLS(x, y)
# print('\n\n')
# 
# ret = [0.065, 0.0265, -0.0593, -0.001,0.0346]
# mktRet = [0.055, -0.09, -0.041,0.045,0.022]
# 
# chp6.linear_least_squares_regression(ret, mktRet)
# print('\n\n')
# 
# sp.random.seed(12456)
# alpha = 1
# beta = 0.8
# n = 100
# x = sp.arange(n)
# y = alpha + beta*x + sp.random.rand(n)
# chp6.linear_least_squares_regression(x, y)
# 
# 
# df = web.DataReader(name='MSFT',
#                     data_source='yahoo')#, start, end, retry_count, pause, session, access_key)
# 
# df2 = web.DataReader(name='^GSPC',
#                     data_source='yahoo')
# 
# msft = pd.DataFrame(df['Adj Close'])
# msft['S&P_Close'] = df2['Adj Close']
# 
# msft['MS_Return'] = msft['Adj Close'].pct_change()
# msft['S&P_Return'] = msft['S&P_Close'].pct_change()
# print(msft)
# 
# x = msft['MS_Return'].dropna()
# y = msft['S&P_Return'].dropna()
# 
# 
# chp6.linear_least_squares_regression(y,x)

"""
Stopped at page 193
"""


"""
#===============================================================================
# Chapter 11
#===============================================================================
"""

# d1 = st.norm.pdf(0)
# print('d1: ', d1)
# 
# 
# d2 = 1/sp.sqrt(2 * sp.pi * 0.05 ** 2) * sp.exp(-(0 - 0.1)**2 / 0.05**2 /2)
# print('d2: ', d2) 


# x = sp.arange(-3,3,0.1)
# y = st.norm.pdf(x)
# plt.title('Std Normal Distribution')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.plot(x,y)
# plt.show()

# concept of VaR based on a
# standard normal distribution with a 95% confidence level

# z = -2.325 # z-normal due to 95% conf. interval
# xStart = -3.8
# yStart = 0.2
# xEnd = -2.5
# yEnd = 0.05
# def f(t):
#     return st.norm.pdf(t)
# 
# plt.ylim(0,0.45)
# x = sp.arange(-3,3,0.1)
# y1 = f(x)
# plt.plot(x,y1)
# x2 = sp.arange(-4,z,1/40)
# sum = 0
# delta = 0.05
# s = sp.arange(-10,z,delta)
# for i in s:
#     sum += f(i)*delta
#     
# plt.annotate('area is '+str(round(sum,4)),xy=(xEnd,yEnd),xytext=(xStart,yStart), arrowprops=dict(facecolor='red',shrink=0.01))
# plt.annotate('z= '+str(z),xy=(z,0.01))
# plt.fill_between(x2,f(x2))
# plt.show()

# df = web.DataReader(name='WMT',
#                     data_source='yahoo',
#                     start='2015-01-01')['Adj Close']
#                     
#                     
# 
# print(df)


# file = r'C:\Users\Marlombrei\Downloads\ff5VWindustryMonthly.pkl'
# ci = 0.99
# z = st.norm.ppf(1- ci)
# x = pd.read_pickle(file)
# 
# print(x.head(),'\n')
# print(x.tail())

##########################################################
# VaR of $1,000 invested in each industry
# position = ([1000]) * 5
# print(position)
# 
# std = sp.std(x, axis=0)
# print(std)
# 
# mean = sp.mean(x, axis=0)
# print(mean)
# 
# t = sp.dot(position,z)
# 
# vv = t * std
# print(vv)

############################################################
import Value_at_Risk as var

# df = web.DataReader(name='WMT',
#                     data_source='yahoo').dropna()#['Adj Close']#, start, end, retry_count, pause, session, access_key)
# 
# 
# ret = df['Adj Close'] / df['Adj Close'].shift(1) - 1 
# ret = ret.dropna()

# print('W-test: ', st.shapiro(ret))
# print('P-value: ', st.anderson(ret))
# 
# print(st.skew(ret))
# print(st.kurtosis(ret))

# price = df['Adj Close'][-1]
# position = 500 * price
# mean_return_daily =  ret.mean()
# std_dev_daily = sp.std(ret, axis=0)
# conf_interval = 0.99 
# period = 1
# 
# print("VaR_parametric:")
# print(var.VaR_parametric(position=position,
#                          mean_return_daily=mean_return_daily,
#                          std_dev_daily=std_dev_daily,
#                          conf_interval=conf_interval,
#                          period=period)) 
# 
# 
# print(-1076.62456084/position,'\n\n')
# 
# 
# print("Modified_VaR")
# print(var.Modified_VaR(position=position,
#                          mean_return_daily=mean_return_daily,
#                          std_dev_daily=std_dev_daily,
#                          conf_interval=conf_interval,
#                          period=period,
#                          ret=ret))
# 
# print(-1554.6058505/position,'\n\n')
# 
# print("Historical VaR")
# print(var.Historical_VaR(position=position,
#                          conf_interval=conf_interval,
#                          ret=ret))
# 
# print(-1338.83202681/position)
# 
# 
# print("Monte-Carlo VaR")
# print(var.MonteCarlo_VaR(position=position,
#                          mean_return_daily=mean_return_daily,
#                          std_dev_daily=std_dev_daily,
#                          conf_interval=conf_interval,
#                          n_sim=5000))


#===============================================================================
"""Calculating Portfolio VaR"""
#===============================================================================
# Step 1
# tickers = ('IBM', 'WMT','C')
# weights = (0.2,0.5,0.3)
# start_date = '2012-1-1'
# end_date = '2016-12-31'
#  
# conf_interval = 0.99
# position = 5e6
# z = st.norm.ppf(1 - conf_interval)
#  
# # Step 2: get the returns dataframe
# prices, ret_f, portReturn, portStd, portMean = var.Portfolio_full(tickers=tickers,
#                                                               weights=weights,
#                                                               start_date=start_date,
#                                                               end_date=end_date)
#  
# portVaR = var.VaR_parametric(position=position,
#                              mean_return_daily=portMean,
#                              std_dev_daily=portStd,
#                              conf_interval=conf_interval,
#                              period=1)
#  
# #print(portVaR)
# # Calculate VaR for each stock
# n = np.size(tickers)
# final = ret_f
# sum_ind_VaR = 0
# for i in np.arange(n):
#     stock = tickers[i]
#     ret = final[stock]
#     position2 = position*weights[i]
#     mean = sp.mean(ret)
#     std = sp.std(ret)
#     ind_VaR = var.VaR_parametric(position=position2,
#                              mean_return_daily=mean,
#                              std_dev_daily=std,
#                              conf_interval=conf_interval,
#                              period=1)
#     sum_ind_VaR += ind_VaR
#     print(ind_VaR)
#  
# print('\n', sum_ind_VaR)



#===============================================================================
"""IBM"""
#===============================================================================
# n_shares = 1000
# ci = 0.99
# period = 1
# start_date = '2016-02-08'
# end_date = '2017-02-07'
# ticker = ('IBM')
# 
# prices, ret_f, shareStd, shareMean = var.Single_shareReturns(ticker=ticker,
#                                                              start_date=start_date,
#                                                              end_date=end_date)
# 
# position = n_shares * prices[-1]
# 
# IBM_VaR = var.VaR_parametric(position=position,
#                              mean_return_daily=shareMean,
#                              std_dev_daily=shareStd,
#                              conf_interval=ci,
#                              period=period)
# 
# print('IBM_VaR: ',IBM_VaR[0])
# print('%IBM_VaR: ', IBM_VaR[0]/position)
# 
# 
# cutOff = IBM_VaR[0]/position # %IBM_VaR
# n = len(ret_f)
# ret2 = ret_f[ret_f <= cutOff].dropna() # filtering values lower than %IBM_VaR
# n2 = len(ret2)
# print('n: ',n)
# print('n2: ',n2)
# print('Ratio: ', n2/n)
# # Historical Expected Shortfall
# print('ES: ', position * (ret_f[ret_f < cutOff].dropna()).mean())
#===============================================================================
# Expected Shortfall
# VaR might underestimate the maximum loss (risk) if we observe a fat tail.
# Expected shortfall(ES) is the expected loss if a VaR is hit
# ES = (loss | z < -alpha)
# Similarly, we could derive the formula to estimate the expected shortfall based on
# historical returns. In a sense, the expected shortfall is the average loss based on
# returns with a lower value than the VaR threshold.

# x = sp.arange(-3,3,0.1)
# ret = st.norm.pdf(x)
# confidence = 0.99
# position = 10000
# z = st.norm.ppf(1 - confidence)
# zES = st.norm.pdf(z)
# std = sp.std(ret)
# VaR = position * z * std
# print('VaR: ',VaR)
# ES = position * zES * std
# print('ES: ', ES)





#zES = -st.norm.pdf(st.norm.ppf(1 - confidence)) / (1 - confidence)
#ES = position * zES * sigma
#===============================================================================


#===============================================================================
# Appendix A - data case 7
#===============================================================================

# tickers = ('MSFT','AAPL','HD','C','WMT','GE')
# tickers = ('MSFT','IBM')
# start_date = '2012-02-07'
# end_date = '2017-02-07'
# position = 1000000
# ci = 0.99
# period = 10
# #weights = (0.17,0.17,0.17,0.17,0.17,0.15)
# weights = (0.25,0.75)
# pos_wei = dict(zip(tickers,weights))
# print(pos_wei)
# 
# for k,v in pos_wei.items():
#     print(k,v)
# prices, ret_f, portReturn, portStd, portMean = var.Portfolio_full(tickers=tickers,
#                                                                    weights=weights,
#                                                                    start_date=start_date,
#                                                                    end_date=end_date)
# 
# VaRs = pd.Series()
# 
# for ticker,pos in pos_wei.items():
#     individual_VaR = var.VaR_parametric(position=position*pos,
#                                            mean_return_daily=ret_f[ticker].mean(),
#                                            std_dev_daily=ret_f[ticker].std(),
#                                            conf_interval=ci,
#                                            period=period)
#     VaRs['VaR_{}: '.format(ticker)] = individual_VaR
#     #print('VaR_{}: '.format(ticker),individual_VaR)  
# 
# print(VaRs)
# 
# VaR_sum = VaRs.sum()
# print(VaR_sum)
# 
# port_VaR = var.VaR_parametric(position=1000000,
#                               mean_return_daily=portMean,
#                               std_dev_daily=portStd,
#                               conf_interval=ci,
#                               period=period)
# 
# print(port_VaR)



#===============================================================================
"""Chapter 12 - MonteCarlo Simulation"""
#===============================================================================


print(var.std_normal_distribution(10))


sp.random.seed(12345)

print(sp.random.normal(loc=0.05,
                       scale=0.1,
                       size=10))



# var.Histograms(bins=10)



# x = sp.random.uniform(low=1, high=100, size=10)
# print(x)
# 
# import random
# def rolldice():
#     
#     roll = random.randint(1,6)
#     return roll
# 
# i = 1
# n = 10
# result = []
# random.seed(123)
# while i < n:
#     result.append(rolldice())
#     i+=1
# 
# print(result)

# Estimating Pi
# n = 10000
# x = sp.random.uniform(low=0, high=1, size=n)
# y = sp.random.uniform(low=0, high=1, size=n)
# 
# dist = sp.sqrt(x**2 + y**2)
# in_circle = dist[dist<=1]
# our_Pi = len(in_circle)*4 / n
# print('pi: {}'.format(our_Pi))
# print('error (%): {}'.format((our_Pi - sp.pi)/sp.pi))


# var.Poisson_Distribution()

n_stocks_available = 500
n_stocks = 10
file = r'C:\Users\Marlombrei\Downloads\yanMonthly.pkl'

x = pd.read_pickle(file)
x2 = sp.unique(np.array(x.index))
# removing indices
x3 = x2[x2 < 'ZZZZ']

sp.random.seed(1234567)
nonStocks = ['GOLDPRICE','HML','SMB','Mkt_Rf','Rf','Russ3000E_D','US_DEBT','Russ3000E_X','US_GDP2009dollar','US_GDP2013dollar']
x4 = list(x3)
for i in range(len(nonStocks)):
    x4.remove(nonStocks[i])

k = sp.random.uniform(low=1,
                      high=len(x4),
                      size=n_stocks)

y,s=[],[]

for i in range(n_stocks):
    index = int(k[i])
    y.append(index)
    s.append(x4[index])
    
final = sp.unique(y)
print(final)
print(s)

print(var.Permutation(x=range(1,11)))






# var.Distribution_of_annual_returns(ticker='msft',
#                                start_date='1986-4-1',
#                                end_date='2013-12-31')

























































































































































































































































































































































































































































