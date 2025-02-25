import pandas as pd
import numpy as np
import functools as ft

def gen_returns_frame(view, days=[1]):
    #view.drop_duplicates('date', keep='first', inplace=True)
    #view.drop([''], axis=1, inplace=True
    view['date'] = pd.to_datetime(view['date'])
    for day in days:
        view[f'{day}_fut_return'] = (view['bid'] + view['ask']) / 2 
        view[f'{day}_fut_return'] = view[f'{day}_fut_return'].pct_change(-day) # return in the future
    return view

def gen_signal(df, time, side):
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'], inplace=True)
    
    if side == 'call':
        df = df[df['cp_flag'] == 'C']
    else:
        df = df[df['cp_flag'] == 'P']
    
    # volume z-score
    avg_vol = df.groupby('date')['volume'].ewm(span=14, adjust=False).mean().groupby('date').mean()
    std_vol = df.groupby('date')['volume'].ewm(span=14, adjust=False).std().groupby('date').mean()
    
    log_volume = np.log(df.groupby('date')['volume'].sum())
    unusual_volume = 1*(df.groupby('date')['volume'].sum() > avg_vol + 0.02*std_vol)*df.groupby('date')['volume'].sum()
    
    # iv top 10%
    iv_change = df.groupby('date').mean('impl_volatility').pct_change().fillna(0)['impl_volatility']
    iv_thres = iv_change.ewm(span=14, adjust=False).mean().quantile(0.9)
    iv_spike = 1*(iv_change > iv_thres)
    
    # weighted delta and normalization
    df.loc[:, 'weight_delta'] = df['delta'] * df['volume']
    delta_change_norm = df.groupby('date')['weight_delta'].sum() / df.groupby('date')['volume'].sum()
    delta_change_thres = 1*(abs(delta_change_norm) > 0.3)
    
    # tight spread check
    #sprd_change = df.groupby('date').apply(lambda x: x['best_offer'] - x['best_bid']).pct_change().fillna(0).groupby('date').mean()
    #tight_sprd = 1*(sprd_change < sprd_change.ewm(span=14).mean().quantile(0.1))  # Bottom 10% of spread changes
    
    
    df_sigs = pd.concat([log_volume, unusual_volume,
                         iv_change, 
                         delta_change_norm], axis=1).dropna()
    df_sigs.columns = [f'log_volume_{time}_{side}',f'unusual_volume_{time}_{side}', f'iv_change_{time}_{side}',
         f'delta_change_norm_{time}_{side}']
    
    return df_sigs

def gen_signal_filter(df):
    df.loc[:, 'DTM'] = (pd.to_datetime(df['exdate']) - pd.to_datetime(df['date'])).dt.days
    
    short_term_df = df[(df['DTM'] < 30)]
    mid_term_df = df[(df['DTM'] > 30) & (df['DTM'] < 180)]
    long_term_df = df[(df['DTM'] > 180)]
    
    df.sig = pd.concat([gen_signal(short_term_df, 'short_term', 'c'), gen_signal(short_term_df, 'short_term', 'p'),
                        gen_signal(mid_term_df, 'mid_term', 'c'), gen_signal(mid_term_df, 'mid_term', 'p'),
                        gen_signal(long_term_df, 'long_term', 'c'), gen_signal(long_term_df, 'long_term', 'p')], axis=1)
    
    return df.sig.fillna(0)


def gen_signals_m(merged_data, equities_data, options_data):
    df_stack = pd.DataFrame()
    for permno in merged_data['permno'].unique():
        stock_view = equities_data[equities_data['permno'] == permno].sort_values('date')
        options_view = options_data[options_data['permno'] == permno].sort_values('date')
    
        stock_view = gen_returns_frame(stock_view, days=[1,2,3,5,7,14,21,28])
        options_sigs = gen_signal_filter(options_view)
    
        series = [stock_view, options_sigs]
    
        df_stack = pd.concat([df_stack, ft.reduce(lambda left, right: pd.merge(left, right, how='left', left_on='date', right_index=True), series)]).fillna(0)

    return df_stack