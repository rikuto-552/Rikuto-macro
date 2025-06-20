import pandas as pd
import numpy as np

pwt1001 = pd.read_stata('https://dataverse.nl/api/access/datafile/354098')

oecd_countries = [
        'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland', 'France',
        'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Japan', 'Netherlands',
        'New Zealand', 'Norway', 'Portugal', 'Spain', 'Sweden', 'Switzerland',
        'United Kingdom', 'United States'
    ]

data = pwt1001[
        pwt1001['country'].isin(oecd_countries) &
        pwt1001['year'].between(1990, 2019)
    ]

relevant_cols = ['countrycode', 'country', 'year', 'rgdpna', 'rkna', 'pop', 'emp', 'avh', 'labsh', 'rtfpna']
data = data[relevant_cols].dropna()

# 追加変数の計算
# alpha: 資本分配率 (1 - 労働分配率)
data['alpha'] = 1 - data['labsh']
# y_n: 労働者一人当たり生産量 (Y/L)
data['y_n'] = data['rgdpna'] / data['emp']
# hours: 総労働時間 (L) (PWTデータでは通常 'emp' と 'avh' で計算可能だが、今回は直接使用しないため影響なし)
data['hours'] = data['emp'] * data['avh']

data = data.sort_values(['country', 'year'])

def calculate_growth_rates_standard(country_data):

    start_year_actual = country_data['year'].min()
    end_year_actual = country_data['year'].max()

    start_data = country_data[country_data['year'] == start_year_actual].iloc[0]
    end_data = country_data[country_data['year'] == end_year_actual].iloc[0]

    years = end_data['year'] - start_data['year']

    # 1. 労働者一人当たり生産量 (Y/L) の年平均成長率 (Growth Rate)
    g_y = ((end_data['y_n'] / start_data['y_n']) ** (1/years) - 1) * 100

    # 2. TFP (A) の年平均成長率 (TFP Growth)
    g_a_true = ((end_data['rtfpna'] / start_data['rtfpna']) ** (1/years) - 1) * 100

    # 3. 労働者一人当たり資本 (K/L) の年平均成長率を計算
    start_kl = start_data['rkna'] / start_data['emp']
    end_kl = end_data['rkna'] / end_data['emp']
    g_kl_true = ((end_kl / start_kl) ** (1/years) - 1) * 100

    # 4. 平均資本分配率 (alpha_avg)
    start_alpha = 1 - start_data['labsh']
    end_alpha = 1 - end_data['labsh']
    alpha_avg = (start_alpha + end_alpha) / 2.0

    # 5. 資本深化の成長への貢献度 (Capital Deepening)
    capital_deepening_contrib_true = alpha_avg * g_kl_true

    # TFP Share と Capital Share の計算
    tfp_share = (g_a_true / g_y)
    cap_share = (capital_deepening_contrib_true / g_y)

    # 結果を辞書形式で返す
    return {
        'Country': start_data['country'],
        'Growth Rate': round(g_y, 2),
        'TFP Growth': round(g_a_true, 2),
        'Capital Deepening': round(capital_deepening_contrib_true, 2),
        'TFP Share': round(tfp_share, 2),
        'Capital Share': round(cap_share, 2)
    }


results_list = data.groupby('country').apply(calculate_growth_rates_standard).dropna().tolist()

results_df = pd.DataFrame(results_list)

avg_row_data = {
    'Country': 'Average',
    'Growth Rate': round(results_df['Growth Rate'].mean(), 2),
    'TFP Growth': round(results_df['TFP Growth'].mean(), 2),
    'Capital Deepening': round(results_df['Capital Deepening'].mean(), 2),
    'TFP Share': round(results_df['TFP Share'].mean(), 2),
    'Capital Share': round(results_df['Capital Share'].mean(), 2)
}
results_df = pd.concat([results_df, pd.DataFrame([avg_row_data])], ignore_index=True)

# 結果を整形して出力
print("\nTable 5.1 Growth Accounting in OECD Countries: 1960-2000")
print("="*85)
print(results_df.to_string(index=False))