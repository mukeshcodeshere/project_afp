import datetime
import random
import gc 
from collections import defaultdict
from datetime import timedelta

from typing import Dict, List, Callable
from joblib import Parallel, delayed

from tqdm import tqdm
import pandas as pd
import numpy as np
import backtrader as bt
import quantstats as qs
import pyfolio as pf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from consts import START_DATE, END_DATE

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

# need to make this dynamic, just add all the data in the df as lines to the feed 
class CustomPandasData(bt.feeds.PandasData):

    # Dont Touch, custom data mapping
    params = (
        ('datetime', None),
        ('open', 'bid'),
        ('high', 'ask'),
        ('low', None),
        ('close', 'prc'),
        ('volume', 'vol'),
        ('openinterest', None),
        ('fut_return_1', '1_fut_return'),
        ('fut_return_2', '2_fut_return'),
        ('fut_return_3', '3_fut_return'),
        ('fut_return_4', '4_fut_return'),
    )


def strategy_quality(
    cer: Callable[[], bt.Cerebro],
    data_source: pd.DataFrame,
    ticker: str,
):
    from_date = data_source.index.min()
    to_date = data_source.index.max()
    days = len(data_source)

    cash = 100000.0
    c = cer()
    c.broker.setcash(cash)
    c.addanalyzer(bt.analyzers.DrawDown)
    
    logging.warning(f"{ticker}")
    
    interval_len = np.random.randint(150, days-30)
    start_day = np.random.randint(0, days - interval_len)
    
    if ticker == 'XOMA':
        print(interval_len)
        print(ticker)
        print(start_day)
        
    start = from_date + datetime.timedelta(days=start_day)
    end = start + datetime.timedelta(days=interval_len)
    data = bt.feeds.PandasData(dataname=data_source.loc[start: end])
    c.adddata(data)
    
    res = c.run()
    val = c.broker.get_value()/cash

    max_dd = res[0].analyzers[0].get_analysis()['max']['drawdown']

    return cer.__name__, ticker, val, max_dd


def evaluate_strategies(
    strategies: List[Callable[[], bt.Cerebro]],
    logs: Dict[str, pd.DataFrame],
    n_trials: int,
    n_jobs: int
):
    tasks = [(s, ln, l) for s in strategies for ln, l in logs.items()]*n_trials

    stats = Parallel(n_jobs)(
        delayed(strategy_quality)(strategy, log, ticker)
        for strategy, ticker, log in tqdm(tasks)
    )

    return pd.DataFrame(stats, columns=['strategy', 'ticker', 'value', 'dropdown'])

# needs modification to be dynamic 
def validate_ticker_data(ticker_data):
    """
    Validate ticker data for required fields and sufficient length
    """
    required_fields = [
        'impl_volatility',
        'put_call_ratio_delta_0.25',
        'open_interest_ratio_delta_0.25'
    ]

    # Check if all required fields are present and non-null
    for field in required_fields:
        if field not in ticker_data.columns or ticker_data[field].isnull().all():
            print(field)
            print(f"Skipping ticker due to missing/null {field}")
            return False

    # Check data length
    if len(ticker_data) < 2:
        print("Insufficient data points")
        return False

    return True

# checks if columns required for strat are present
def preprocess_dataframe(df_stack):
    """
    Robust preprocessing of input DataFrame
    """
    try:
        # Convert date and drop invalid rows
        df_stack['date'] = pd.to_datetime(df_stack['date'], errors='coerce')
        df = df_stack.dropna(subset=['date'])

        # Verify required columns
        required_columns = [
            'permno', 'bid', 'ask', 'prc', 'vol', 'impl_volatility',
            'put_call_ratio_delta_0.25', 'open_interest_ratio_delta_0.25'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df
    except Exception as e:
        print(f"Error during DataFrame preprocessing: {e}")
        raise


def analyze_portfolio_returns(portfolio_returns, strategy_name):
    """
    Analyze and visualize portfolio returns with key performance metrics.

    Parameters:
    portfolio_returns (OrderedDict): Dictionary of datetime to returns
    strategy_name (str): Name of the strategy for differentiation

    Returns:
    dict: Performance metrics
    """
    # Convert OrderedDict to pandas Series
    returns_series = pd.Series(portfolio_returns)
    returns_series.index = pd.to_datetime(returns_series.index)

    # Calculate cumulative returns
    cumulative_returns = (1 + returns_series).cumprod() - 1

    # Calculate annualized return
    trading_days = len(returns_series)
    years = trading_days / 252  # Assuming 252 trading days per year
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Calculate Sharpe Ratio (assuming risk-free rate of 0)
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    sharpe_ratio = mean_return / std_return * np.sqrt(252)

    # Calculate Sortino Ratio
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std != 0 else 0

    # Calculate Drawdown
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()

    # Create visualizations

    # Plot Cumulative Returns
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.xlabel(f'Date ({strategy_name})')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.savefig(f"cumulative_returns_{strategy_name}.png")
    plt.show()

    # Plot Daily Returns
    plt.figure(figsize=(10, 6))
    returns_series.plot(kind='bar')
    plt.xlabel(f'Date ({strategy_name})')
    plt.ylabel('Daily Return')
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(f"daily_returns_{strategy_name}.png")
    plt.show()

    # Plot Drawdown
    plt.figure(figsize=(10, 6))
    drawdown.plot()
    plt.xlabel(f'Date ({strategy_name})')
    plt.ylabel('Drawdown')
    plt.grid()
    plt.savefig(f"drawdown_{strategy_name}.png")
    plt.show()

    # Plot Histogram of Returns
    plt.figure(figsize=(10, 6))
    returns_series.hist(bins=30)
    plt.xlabel(f'Return ({strategy_name})')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(f"histogram_returns_{strategy_name}.png")
    plt.show()

    # Prepare metrics dictionary
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

    return {
        'metrics': metrics
    }
        

# def run_backtest(df, strategies):
#     """
#     Run backtest for given strategies
#     """
#     # Get unique tickers
#     unique_tickers = df['permno'].unique()

#     for strategy_name, strategy_class in strategies:
#         cerebro = bt.Cerebro()
#         cerebro.addstrategy(strategy_class)

#         # Track valid tickers
#         valid_tickers_added = 0

#         # Add data feeds for each ticker
#         for ticker in unique_tickers:
#             try:
#                 ticker_data = df[df['permno'] == ticker].copy()

#                 # Robust date handling
#                 ticker_data['date'] = pd.to_datetime(ticker_data['date'], errors='coerce')
#                 ticker_data = ticker_data.sort_values('date')
#                 ticker_data.set_index('date', inplace=True)

#                 # Validate ticker data
#                 if not validate_ticker_data(ticker_data):
#                     print(f"Skipping invalid ticker: {ticker}")
#                     continue

#                 data_feed = CustomPandasData(dataname=ticker_data)
#                 cerebro.adddata(data_feed, name=str(ticker))
#                 valid_tickers_added += 1

#             except Exception as e:
#                 print(f"Error processing ticker {ticker}: {e}")
#                 continue

#         # Skip backtest if no valid tickers
#         if valid_tickers_added == 0:
#             print(f"No valid tickers for strategy {strategy_name}")
#             continue

#         # Configure backtest
#         cerebro.broker.set_cash(100000)
#         cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

#         try:
#             # Run backtest
#             results = cerebro.run(maxcpus=4)
#             strategy = results[0]

#             # Extract portfolio returns
#             portfolio_returns = strategy.analyzers.pyfolio.get_analysis()['returns']

#             if portfolio_returns:
#                 returns_series = pd.Series(list(portfolio_returns.values()), index=list(portfolio_returns.keys()))
#                 analyze_portfolio_returns(returns_series, strategy_name)

#         except Exception as e:
#             print(f"Backtest error for {strategy_name}")
#             traceback.print_exc()

#         finally:
#             gc.collect()

def calc_train_test_days(start, end, verbose=False):
    # Define time periods
    total_days = (end - start).days + 1

    # Calculate split days
    train_days = int(total_days * 0.6)
    validation_days = int(total_days * 0.2)
    test_days = total_days - train_days - validation_days

    # Calculate split dates
    train_start = start
    train_end = train_start + timedelta(days=train_days - 1)

    validation_start = train_end + timedelta(days=1)
    validation_end = validation_start + timedelta(days=validation_days - 1)

    test_start = validation_end + timedelta(days=1)
    test_end = end
    
    if verbose:
        print(f"Total Days: {total_days}")
        print(f"\nTrain Period:")
        print(f"Start: {train_start.strftime('%Y-%m-%d')}")
        print(f"End: {train_end.strftime('%Y-%m-%d')}")
        print(f"Days: {train_days}")
    
        print(f"\nValidation Period:")
        print(f"Start: {validation_start.strftime('%Y-%m-%d')}")
        print(f"End: {validation_end.strftime('%Y-%m-%d')}")
        print(f"Days: {validation_days}")
    
        print(f"\nTest Period:")
        print(f"Start: {test_start.strftime('%Y-%m-%d')}")
        print(f"End: {test_end.strftime('%Y-%m-%d')}")
        print(f"Days: {test_days}")

    return train_start, train_end, validation_start, validation_end, test_start, test_end


def train_models(data_dict, features, train_start, train_end):
    """ 
    given train period trains a linear regression to fit to the features,
     combinging into a single signal
    """
    
    models = {}

    for permno, data in data_dict.items():
        train_data = data.loc[train_start:train_end]
        if len(train_data) > 0:
            X_train = train_data[features]
            y_train = train_data['target']  # Ensure 'target' is defined in your data preparation step

            model = LinearRegression()
            model.fit(X_train, y_train)
            models[permno] = model

    return models


def generate_signals(models, features, data_dict, test_start, test_end):
    """
    using trained models generates the investment signals for the test set 
    """
    signals_dict = {}

    for permno, model in models.items():
        test_data = data_dict[permno].loc[test_start:test_end]
        if len(test_data) > 0:
            X_test = test_data[features]
            signals = pd.Series(model.predict(X_test), index=test_data.index)
            signals_dict[permno] = signals

    return signals_dict


def run_backtest(data_dict, signals_dict, strategy, strategy_feed, test_start, test_end, initial_capital=1_000_000):
    "runs and analyzes backtest on individual stocks in the dataset"
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)

    for permno, data in data_dict.items():
        if permno in signals_dict:
            test_data = data.loc[test_start:test_end].copy()
            test_data['predicted_signal'] = signals_dict[permno]
            data_feed = strategy_feed(
                dataname=test_data,
                name=str(permno),  # Convert permno to string for naming
                fromdate=pd.to_datetime(test_start),
                todate=pd.to_datetime(test_end)
            )
            cerebro.adddata(data_feed)

    cerebro.addstrategy(strategy)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    print(f'Starting Portfolio Value: ${initial_capital:,.2f}')
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: ${final_value:,.2f}')

    return results[0], cerebro  # Return both results and cerebro


def main(data, features, strategy, strategy_feed, verbose=False):

    """
    Using the global start and end days splits data into train, validation, and testing set.
    Model is trained on the signal set, then used to invest. 

    data: df of equities, options, and signals data for all stocks
    features: columns of data to use to train regression for final signal
    strategy: cerebero strategy class 
    strategy feed: data feed object which configues lines and additional data

    """

    (train_start, 
     train_end, 
     validation_start, 
     validation_end, 
     test_start, 
     test_end) = calc_train_test_days(START_DATE, END_DATE, verbose)

    # Prepare data for each stock
    data_dict = {}
    for permno, group in data.groupby('permno'):
        group['date'] = pd.to_datetime(group['date'])
        group.set_index('date', inplace=True)
        data_dict[permno] = group

    # Check data preparation
    print("Checking data preparation...")
    print(data.head())
    print(data.isnull().sum())
    print(data.index)

    # Train models using training data
    print("Training models...")
    models = train_models(data_dict, features, train_start, train_end)

    # Validate models
    print("\nValidating models...")
    validation_signals = generate_signals(models, features, data_dict, validation_start, validation_end)

    # Check signal generation
    if verbose:
        print("Checking signal generation...")
        for permno, signals in validation_signals.items():
            print(f"Signals for {permno}:")
            print(signals.head())
            print(signals.describe())

    # Check data feed
    print("Checking data feed...")
    for permno, data in data_dict.items():
        if permno in validation_signals:
            test_data = data.loc[validation_start:validation_end].copy()
            test_data['predicted_signal'] = validation_signals[permno]
            if verbose:
                print(f"Data feed for {permno}:")
                print(test_data[['predicted_signal', 'impl_volatility', 'put_call_ratio_delta_0_25', 'open_interest_ratio_delta_0_25']].head())

    # Run backtest
    print("Running backtest...")
    validation_results, cerebro = run_backtest(data_dict, 
                                               validation_signals, 
                                               strategy,
                                               strategy_feed,
                                               validation_start,
                                               validation_end)

    # Check broker and portfolio value
    print("Checking broker and portfolio value...")
    print(f"Initial Capital: ${cerebro.broker.getvalue():,.2f}")

    
    # Test Final Performance
    print("\nTesting final performance...")
    test_signals = generate_signals(models, features, data_dict, test_start, test_end)
    test_results, cerebro = run_backtest(data_dict,
                                         test_signals, 
                                         strategy, 
                                         strategy_feed,
                                         test_start, 
                                         test_end)

    # Analyze results
    analyzer = test_results.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = analyzer.get_pf_items()

    # Plot performance metrics
    plt.figure(figsize=(15, 10))
    pf.plot_returns(returns)
    plt.title('Strategy Returns Over Time')
    plt.tight_layout()
    plt.savefig('strategy_returns.png')

    # Plot performance metrics
    plt.figure(figsize=(15, 10))
    pf.plot_returns(np.cumprod(returns + 1) - 1)
    plt.title('Strategy Cumulative Returns Over Time')
    plt.tight_layout()
    plt.savefig('strategy_cumreturns.png')

    # Generate performance statistics
    stats = pf.timeseries.perf_stats(returns)
    print("\nStrategy Performance Statistics:")
    print(stats)

    return models, test_results, stats, [analyzer.get_pf_items(), test_signals]


    

    