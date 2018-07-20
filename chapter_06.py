# from __future__ import division, print_function
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
# import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=500)


def linear_least_squares_regression(x, y):
    """Calculate a linear least-squares regression for two sets of measurements"""
    (beta, alpha, r_value, p_value, std_err) = stats.linregress(x, y)
    print('Beta: ', beta)  # Slope
    print('Alpha: ', alpha)  # Intercept
    print('R-squared: ', r_value ** 2)
    print('P-value: ', p_value)
    print('Std-Error: ', std_err)


def OLS(x, y):
    x = sm.add_constant(x)
    results = sm.OLS(y, x).fit()
    print(results.params)
    print(results.summary())

