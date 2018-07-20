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


def Rc2Rm(Rc, m):
    return np.exp(Rc / m) - 1


def pandas_interpolate(times, rates):
    """Interpolating rates using Pandas
    Function will need improvement for shorter time periods"""
    x = pd.Series(rates, index=times)
    y = pd.Series(np.nan, index=range(1, max(times) + 1))
    curves = y.align(x)  # it returns a tuple
    return curves[1].interpolate()


def YTM_zero_cupon(FV, PV, n):
    """
    The term structure of interest rates is very important since it serves as a benchmark
    to estimate Yield to Maturity (YTM) for corporate bonds. YTM is the period return
    if the bond holder holds until the bond expires. Technically speaking, YTM is the
    same as Internal Rate of Return (IRR). In the financial industry, the spread, defined
    as the difference between YTM of a corporate bond over the risk-free rate, is used to
    estimate the discount rate for corporate bonds. The spread is a measure of the default
    risk. Thus, it should be closely correlated with the credit rating of the company and
    of the bond.
    """
    return (FV / PV) ** (1 / n) - 1


def YTM(time, dollar_coupon, pv, face_vale):
    return sp.rate(time, dollar_coupon, pv, face_vale)   


def duration(t, cash_flow, y):
    """
    The durationis defined as: the number of years needed
    to recover our initial investment.
    n is the number of cash flows
    wi is the weight of the ith cash flow
    and wi is defined as the present value of ith cash flow over the present values of all cash flows
    Ti is the timing (in years) of the ith cash flow
    y is the YTM
    """
    n = len(t)
    B = 0  # B is the bond's present value
    for i in range(n):
        B += cash_flow[i] * np.exp(-y * t[i])
    
    D = 0  # D is the duration
    for i in range(n):
        D += t[i] * cash_flow[i] * np.exp(-y * t[i]) / B
    return D


def durationBond(rate, couponRate, maturity):
    """Objective : estimte the durtion for a given bond
    rate : discount rate
    couponRate: coupon rate
    maturity : number of years
    Example 1: >>>discountRate=0.1
    >>>couponRate=0.04
    >>> n=4
    >>>durationBond(rate,couponRate,n)
    3.5616941835365492
    Example #2>>>durationBond(0.1,0.04,4)
    3.7465335177625576
    """
    d = 0
    n = maturity
    for i in sp.arange(n):
        d += (i + 1) * sp.pv(rate, i + 1, 0, -couponRate)
        d += n * sp.pv(rate, n, 0, -1)
    return d / sp.pv(rate, n, -couponRate, -1)

