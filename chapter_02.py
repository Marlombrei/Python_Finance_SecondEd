# from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats as st
import pandas as pd
import pandas_datareader.data as web
import pandas.tseries
import statsmodels.api as sm
# from datetime import datetime, date, time
# from dateutil.parser import parse
# import csv
# import json
# import openpyxl
# from openpyxl import load_workbook
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=500)

"""
ordinary least square (OLS) regression is a method for
estimating the unknown parameters in a linear regression model. It minimizes the sum
of squared vertical distances between the observed values and the values predicted
by the linear approximation.
"""

# y is an n by 1 vector (array)
# is an n by (m+1) matrix
# return matrix (n by m), plus a vector that contains 1 only
#    n is the number of observations, and m is the number of independent variables

# # OLS
# y = [1,2,3,4,2,3,4]
# x = range(1,8)
# x = sm.add_constant(x)
# results = sm.OLS(y,x).fit()
# print(results.params)
# print('The intercept is: {}'.format(results.params[0]))
# print('The slope is: {}'.format(results.params[1]))

# # Interpolation
# #np.random.seed(123)
# x = np.arange(1,10.1,0.25) ** 2
# n = np.size(x)
# y = pd.Series(x + np.random.randn(n))
# bad = np.array([4,13,14,15,16,20,30])
# y[bad] = np.nan
# print(y)
# 
# methods = ['linear', 'quadratic', 'cubic']
# df = pd.DataFrame({m: y.interpolate(method=m) for m in methods})
# df.plot()
# plt.show()

# df = web.get_data_google('ibm')

