import pandas as pd
import functools as ft

def gen_returns_frame(view, days=[1]):
    #view.drop_duplicates('date', keep='first', inplace=True)
    #view.drop([''], axis=1, inplace=True
    for day in days:
        view[f'{day}_fut_return'] = (view['bid'] + view['ask']) / 2
        view[f'{day}_fut_return'] = view[f'{day}_fut_return'].pct_change(-day) # return in the future
    return view


def gen_norm_option_stats(view):
    view = view.groupby('date').mean('impl_volatility').fillna(0)
    return view['impl_volatility']


def merge_put_call(calls, puts, greek, value):
    df_sig = pd.merge(calls[['volume', 'open_interest']], puts[['volume', 'open_interest']], how='left', left_on='date', right_on='date', suffixes=('_call', '_put'))#['volume_put']
    df_sig = df_sig.fillna(0)
    df_sig[f'put_call_ratio_{greek}_{value}'] = df_sig['volume_put'] / (df_sig['volume_call'] + df_sig['volume_put'])
    df_sig[f'open_interest_ratio_{greek}_{value}'] = df_sig['open_interest_put'] / (df_sig['open_interest_call'] + df_sig['open_interest_put'])
    return df_sig[[f'put_call_ratio_{greek}_{value}', f'open_interest_ratio_{greek}_{value}']]


### Signals of interest
def gen_put_call_ratio(view):
    # filtered for low delta values
    # delta sigs
    calls = view[(view['cp_flag'] == 'C') & (view['delta'] < 0.15)].groupby('date').sum('volume')
    puts = view[(view['cp_flag'] == 'P') & (view['delta'] < -0.15)].groupby('date').sum('volume')

    delta = merge_put_call(calls, puts, 'delta', 0.15).fillna(0)

    calls = view[(view['cp_flag'] == 'C') & (view['delta'] < 0.25)].groupby('date').sum('volume')
    puts = view[(view['cp_flag'] == 'P') & (view['delta'] < -0.25)].groupby('date').sum('volume')

    delta1 = merge_put_call(calls, puts, 'delta', 0.25).fillna(0)

    # gamma sigs
    calls = view[(view['cp_flag'] == 'C') & (view['gamma'] > 0.20)].groupby('date').sum()
    puts = view[(view['cp_flag'] == 'P') & (view['gamma'] > 0.20)].groupby('date').sum()

    gamma = merge_put_call(calls, puts, 'gamma', 0.2)

    calls = view[(view['cp_flag'] == 'C') & (view['gamma'] > 0.10)].groupby('date').sum()
    puts = view[(view['cp_flag'] == 'P') & (view['gamma'] > 0.10)].groupby('date').sum()

    gamma1 = merge_put_call(calls, puts, 'gamma', 0.1)

    # theta sigs
    calls = view[(view['cp_flag'] == 'C') & (view['theta'] < -25)].groupby('date').sum()
    puts = view[(view['cp_flag'] == 'P') & (view['theta'] < -25)].groupby('date').sum()

    theta = merge_put_call(calls, puts, 'theta', 25)

    calls = view[(view['cp_flag'] == 'C') & (view['theta'] < -15)].groupby('date').sum()
    puts = view[(view['cp_flag'] == 'P') & (view['theta'] < -15)].groupby('date').sum()

    theta1 = merge_put_call(calls, puts, 'theta', 15)

    df_sigs = pd.concat([delta, delta1, gamma, gamma1, theta, theta1], axis=1)

    return df_sigs.fillna(0)

### Generate Signal Dataset
def gen_signals_l(merged_data, equities_data, options_data):
    df_stack = pd.DataFrame()
    for permno in merged_data['permno'].unique():
      # for each stock generate the respective metrics, then stack into panel format
        stock_view = equities_data[equities_data['permno'] == permno].sort_values('date')
        options_view = options_data[options_data['permno'] == permno].sort_values('date')
    
        stock_view = gen_returns_frame(stock_view, days=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        options_metrics = gen_norm_option_stats(options_view)
        options_sigs = gen_put_call_ratio(options_view)
    
        series = [stock_view, options_metrics, options_sigs] # order matters!!!
    
        # stacks individual stocks
        #df_stack = pd.concat([df_stack, pd.merge(stock_view, options_view, how='left', left_on='date', right_index=True)])
        df_stack = pd.concat([df_stack, ft.reduce(lambda left, right: pd.merge(left, right, how='left', left_on='date', right_index=True), series)]).fillna(0)

    return df_stack
