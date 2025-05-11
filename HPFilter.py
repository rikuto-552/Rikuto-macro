import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1955-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
gdp = web.DataReader('CLVMNACSCAB1GQFR', 'fred', start_date, end_date)
log_gdp = np.log(gdp)

#λ値を設定
lamdas=[10, 100, 1600]

#トレンド・循環成分を格納
trends = {}
cycles = {}

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
for lam in lamdas:
    cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=lam)
    trends[lam] = trend
    cycles[lam] = cycle

# 6. グラフ1：元データとトレンド成分の比較
plt.figure(figsize=(12, 6))
plt.plot(log_gdp, label="Original GDP (in log)")
plt.plot(trends[10], label="Trend (λ=10)")
plt.plot(trends[100], label="Trend (λ=100)")
plt.plot(trends[1600], label="Trend (λ=1600)")
plt.legend()
plt.title("France GDP: Trends with Different Lambda Values")
plt.xlabel("Date")
plt.ylabel("Log GDP")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. グラフ2：循環成分の比較
plt.figure(figsize=(12, 6))
plt.plot(cycles[10], label="Cycle (λ=10)")
plt.plot(cycles[100], label="Cycle (λ=100)")
plt.plot(cycles[1600], label="Cycle (λ=1600)")
plt.legend()
plt.title("France GDP: Cyclical Components with Different Lambda Values")
plt.xlabel("Date")
plt.ylabel("Cyclical Component")
plt.grid(True)
plt.tight_layout()
plt.show()