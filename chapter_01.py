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

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=500)


def simple_npv(rate, cash_flows):
    total = 0
    for i in range(len(cash_flows)):
        total += cash_flows[i] / (1 + rate) ** i
    return total


def simple_npv_enumerate(rate, cash_flows):
    total = 0
    for i, cashflow in enumerate(cash_flows):
        total += cashflow / (1 + rate) ** i
    return total

