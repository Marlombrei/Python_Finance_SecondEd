from __future__ import division
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats as st
import pandas as pd
import pandas_datareader.data as web
import pandas.tseries
import matplotlib.pyplot as plt



pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 100
np.set_printoptions(linewidth=500)

def std_normal_distribution(t):
    """ standard normal distribution
        probability density function """
    return st.norm.pdf(t)

def logNormal_distribution(mean=0, sigma=1, size=1000):
    """When stock returns follow a normal distribution, then its prices should follow a
        lognormal distribution.
    density of a lognormal distribution"""
    return sp.random.lognormal(mean, sigma,size)
    
    

def VaR_parametric(position, mean_return_daily, std_dev_daily, conf_interval, period):
    # position: the current market value of our portfolio or position
    # mean_return: the expected period return
    # conf_interval: unded to calculate z-score
    # period is the period of conversion i.e. from daily to monthly
    mean_ret_period = (mean_return_daily + 1)**period - 1
    sigma_period = std_dev_daily * np.sqrt(period)
    z = st.norm.ppf(1 - conf_interval)
    var = position * (mean_ret_period + z*sigma_period)
    return var


def Modified_VaR(position, mean_return_daily, period, std_dev_daily, conf_interval, ret):
    """This is the equivalent of Corner-Fisher VaR"""
    z = abs(st.norm.ppf(1 - conf_interval))
    S = st.skew(ret)
    K = st.kurtosis(ret)
    t = (z * 1/6*(z**2 - 1)*S
         + 1/24*(z**3 - 3*z)*K
         - 1/36*(2*z**3 - 5*z)*S**2)
    
    mean_ret_period = (mean_return_daily + 1)**period - 1
    sigma_period = std_dev_daily * np.sqrt(period)
    mVaR = position * (mean_ret_period - t*sigma_period)
    return mVaR
    

def Historical_VaR(position,conf_interval,ret):
    n = len(ret)
    t = int(n * (1 - conf_interval))
    var = position * ret.sort_values()[t] # not sure if I have to subtract by 1 to the t-th element due to Series starting at 0
    return var


def MonteCarlo_VaR(position, mean_return_daily, std_dev_daily, conf_interval, n_sim):
    """n_sim : number of simulations"""
    sp.random.seed(12345)
    # Monte-Carlo Simulation - Vectorized
    ret2 = sp.random.normal(mean_return_daily, std_dev_daily, n_sim)
    ret3 = np.sort(ret2)
    m = int(n_sim * (1 - conf_interval))
    var = position * ret3[m]
    return var


def Single_shareReturns(ticker, start_date, end_date):
    # Getting all stocks prices
    prices = web.DataReader(name=ticker,
                            data_source='yahoo',
                            start=start_date,
                            end=end_date).dropna()['Adj Close']

    # Calculating the returns for each stock                    
    ret_f = pd.DataFrame()
    ret_f[ticker] = prices / prices.shift(1) - 1    
    ret_f.dropna(inplace=True)

    shareStd = sp.std(ret_f)
    shareMean = sp.mean(ret_f)
    return (prices, ret_f, shareStd, shareMean)
  

def Portfolio_full(tickers, weights, start_date, end_date):
    # Getting all stocks prices
    prices = pd.DataFrame()
    for ticker in tickers:
        prices[ticker] = web.DataReader(name=ticker,
                            data_source='yahoo',
                            start=start_date,
                            end=end_date).dropna()['Adj Close']

    # Calculating the returns for each stock                    
    ret_f = pd.DataFrame()
    for ticker in tickers:
        #price = prices[ticker]
        ret_f[ticker] = prices[ticker] / prices[ticker].shift(1) - 1
         
    ret_f.dropna(inplace=True)
    portReturn = sp.dot(ret_f, weights)
    portStd = sp.std(portReturn)
    portMean = sp.mean(portReturn)
    return (prices, ret_f, portReturn, portStd, portMean)


def Portfolio_VaR():
    pass


def Histograms(mean=0, std=1, n=1000, bins=15):
    x = sp.random.normal(mean, std, n)
    plt.hist(x, bins=bins, normed=True)
    plt.title("Histogram for random numbers drawn from a normal distribution")
    plt.annotate("mean="+str(mean),xy=(2.5,0.3))
    plt.annotate("std="+str(std),xy=(2.5,0.28))
    plt.show()


def Poisson_Distribution():
    x = sp.random.poisson(lam=1, size=100)
    a = 5 #shape
    n = 1000
    s = np.random.power(a,n)
    count, bins, ignored = plt.hist(s, bins=30)
    x = np.linspace(0,1,100)
    y = a*x**(a-1)
    normed_y = n*np.diff(bins)[0] * y
    plt.title('Poisson Distribution')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot(x, normed_y)
    plt.show()

def Permutation(x):
    y = np.random.permutation(x)
    return y


def bootstrap_f(data, num_obs, replacement=None):
    """
    The constraint specified in the previous program is that the number of given
    observations should be larger than the number of random returns we plan to pick
    up. This is true for the bootstrapping without the replacement method. For the
    bootstrapping with the replacement method, we could relax this constraint
    """
    n = len(data)
    if n < num_obs:
        print('n is less than n_obs')
    else:
        if replacement==None:
            y = np.random.permutation(data)
            return y
        else:
            y = []
    
    for i in range(num_obs):
        k = np.random.permutation(data)
        y.append(k[0])
    return y

def Distribution_of_annual_returns(ticker, start_date, end_date, n_simulation=5000):
    """
    estimate its daily returns and convert them into annual
    ones. Based on those annual returns, we generate its distribution by applying
    bootstrapping with replacements 5,000 times
    """
    msft_full = web.DataReader(name=ticker,
                                data_source='yahoo',
                                start=start_date,
                                end=end_date).dropna()#['Adj Close']

    msft_full['logret'] = np.log(msft_full['Adj Close'] / msft_full['Adj Close'].shift(1))
    
    msft = msft_full[['Adj Close','logret']].copy()
    msft = msft[(msft != 0).all(1)]
    
    """ *** Most useful part of the code *** """
    ret_annual = np.exp(msft['logret'].groupby((msft.index.year)).apply(sum))-1
    
    
    n_obs = len(ret_annual)
    np.random.seed(123577)
    final = np.zeros(n_obs,dtype=float)
    for i in range(0,n_obs):
        x = np.random.uniform(low=0,
                              high=n_obs,
                              size=n_obs)
        y = []
        for j in range(n_obs):
            y.append(int(x[j]))
            z = np.array(ret_annual)[y]
        final[i] = np.mean(z)
    mean_annual=round(np.mean(np.array(ret_annual)),4)
    plt.figtext(0.63,0.8,'mean annual='+str(mean_annual))
    plt.hist(final, 50, normed=True)
    plt.show()



































































































































