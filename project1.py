import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np # pandas_datareader as pdr

#1 France

# set the start and end dates for the data
start_date = '1955-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
gdp_france = web.DataReader('CLVMNACSCAB1GQFR', 'fred', start_date, end_date)
log_gdp_france = np.log(gdp_france)

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
cycle_france_np, trend_france = sm.tsa.filters.hpfilter(log_gdp_france, lamb=1600)

# IMPORTANT FIX: Convert NumPy array to pandas Series, preserving original index and naming the series
cycle_france = pd.Series(cycle_france_np, index=log_gdp_france.index, name='France_Cycle')

# Calculate the standard deviation of the cyclical component
std_dev_cycle_france = np.std(cycle_france) 

print(f"フランスの循環成分の標準偏差: {std_dev_cycle_france:.4f}")

#2 Japan

# set the start and end dates for the data
start_date = '1955-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
gdp_japan = web.DataReader('JPNNGDP', 'fred', start_date, end_date)
log_gdp_japan = np.log(gdp_japan)

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
cycle_japan_np, trend_japan = sm.tsa.filters.hpfilter(log_gdp_japan, lamb=1600)

# IMPORTANT FIX: Convert NumPy array to pandas Series, preserving original index and naming the series
cycle_japan = pd.Series(cycle_japan_np, index=log_gdp_japan.index, name='Japan_Cycle')


# Calculate the standard deviation of the cyclical component
std_dev_cycle_japan = np.std(cycle_japan)

print(f"日本の循環成分の標準偏差: {std_dev_cycle_japan:.4f}")

# --- 3. フランスと日本の循環変動成分の相関係数を計算 ---
print("\n# --- フランスと日本の循環変動成分の相関係数の計算 ---")

# 両国の循環成分のSeriesをDataFrameに結合する

combined_cycles = pd.concat([cycle_france, cycle_japan], axis=1)

# NaNを含む行を削除することで、両国にデータが存在する共通の期間だけを残す
aligned_cycles = combined_cycles.dropna()

# 相関係数を計算する
correlation_coefficient = aligned_cycles['France_Cycle'].corr(aligned_cycles['Japan_Cycle'])

print(f"フランスと日本の循環変動成分の相関係数: {correlation_coefficient:.4f}")
print(f"相関係数計算に使用されたデータ期間: {aligned_cycles.index.min().strftime('%Y-%m-%d')} から {aligned_cycles.index.max().strftime('%Y-%m-%d')}")
