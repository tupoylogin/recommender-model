import json
import requests
import typing as tp

import pandas as pd
import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer(object):
    def __init__(self, stocks: tp.List[str],
        days_ago_to_fetch: int,
        ) -> None:

        self._coins = stocks
        self._coin_history = self._fetch_all(stocks, days_ago_to_fetch)

    @staticmethod
    def _fetch_all(stocks: tp.List[str], history: int):
        coin_history = {}

        def index_history(hist):
            # index by date so we can easily filter by a given timeframe
            hist = hist.set_index('time')
            hist.index = pd.to_datetime(hist.index, unit='s')
            return hist

        def filter_history_by_date(hist):
            result = hist[hist.index.year >= 2019]
            # result = result[result.index.day == 1] # every first of month, etc.
            return result

        def fetch_history(coin):
            endpoint_url = "https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym=USD&limit={:d}".format(coin, history)
            res = requests.get(endpoint_url)
            hist = pd.DataFrame(json.loads(res.content)['Data'])
            hist = index_history(hist)
            hist = filter_history_by_date(hist)
            return hist

        for coin in stocks:
            coin_history[coin] = fetch_history(coin)
        return coin_history

    @staticmethod
    def get_all_returns(stocks: tp.List[str], history: pd.DataFrame):
        average_returns = {}
        cumulative_returns = {}
        coin_history = history.copy()
        for coin in stocks:
            hist = coin_history[coin]
            hist['return'] = (hist['close'] - hist['open']) / hist['open']
            average = hist["return"].mean()
            average_returns[coin] = average
            cumulative_returns[coin] = (hist["return"] + 1).prod() - 1
            hist['excess_return'] = hist['return'] - average
            coin_history[coin] = hist
        return coin_history, average_returns, cumulative_returns
    
    @staticmethod
    def get_excess_matrix(hist_length: int, history: pd.DataFrame, coins: tp.List[str]):
        excess_matrix = np.zeros((hist_length, len(coins)))
        coin_history = history.copy()

        for i in range(0, hist_length):
            for idx, coin in enumerate(coins):
                excess_matrix[i][idx] = coin_history[coin].iloc[i]['excess_return']
        
        return excess_matrix
    
    @staticmethod
    def get_varcovar_matrix(excess_matrix: np.ndarray, hist_length: int):
        product_matrix = np.matmul(excess_matrix.transpose(), excess_matrix)
        var_covar_matrix = product_matrix / hist_length
        return var_covar_matrix
    
    @staticmethod
    def get_stddev_matrix(coins: tp.List[str], history: pd.DataFrame):
        coin_history = history.copy()
        std_deviations = np.zeros((len(coins), 1))
        for idx, coin in enumerate(coins):
            std_deviations[idx][0] = np.std(coin_history[coin]['return'])
        return std_deviations
    
    @staticmethod
    def get_corr_matrix(std_deviations: np.ndarray, varcovar: np.ndarray):
        sdev_product = np.matmul(std_deviations, std_deviations.T)
        correlation_matrix = varcovar / sdev_product
        return correlation_matrix
    
    def train_test_split(self):
        train, test = {}, {}
        for k, v in self._coin_history.items():
            train.update({k: v.iloc[:-100, :]})
            test.update({k: v.iloc[-100:, :]})
        return train, test
    
    @staticmethod
    def get_asset_shares(weights, history):
        asset = pd.DataFrame(index=history[list(weights.keys())[0]].index)
        asset['close'] = np.sum([(history[k]['close']*v).values for k,v in weights.items()], axis=0)
        return asset


    
    def optimize(self, desired_annualy_rate: float):
        mu = 1+(desired_annualy_rate/252)
        train, test = self.train_test_split()
        print(train)
        hist_len = len(train[self._coins[0]])
        history, average_returns, cumulative_returns = self.get_all_returns(self._coins, train)
        std_deviations = self.get_stddev_matrix(self._coins, history)
        excess_matrix = self.get_excess_matrix(hist_len, history, self._coins)
        var_covar_matrix = self.get_varcovar_matrix(excess_matrix, hist_len)
        correlation_matrix = self.get_corr_matrix(std_deviations, var_covar_matrix)
        coins = self._coins
        run = True

        def volatility(coin_weights: np.ndarray):
            weighted_std_devs = np.multiply(coin_weights, std_deviations)
            product_1 = weighted_std_devs.T
            product_2 = np.matmul(product_1, correlation_matrix)
            portfolio_variance = np.matmul(product_2, weighted_std_devs)
            portfolio_volatility = np.sqrt(np.sum(portfolio_variance))
            return portfolio_volatility

        def sharpe(coin_weights: np.ndarray):
            portfolio_volatility = volatility(coin_weights)
            portfolio_return = np.sum(np.multiply(coin_weights, returns))   
            sharpe_ratio = portfolio_return/portfolio_volatility
            return sharpe_ratio

        while run:

            returns = np.full((len(coins), 1), 0.0)

            zero_constraint = tuple([(-1e3, None) for _ in range(len(coins))])

            constraints = ({'type': 'eq',
                'fun' : lambda x: np.dot(x, np.ones(len(coins))) - 1.,
                'jac' : lambda x: np.ones(len(coins))},
                        {'type': 'ineq',
                'fun' : lambda x: np.dot(x, returns) - mu,
                'jac' : lambda x: np.ones(len(coins))})
            x0 = np.random.random(len(coins))

            res = minimize(lambda x: -1*sharpe(x), x0, method='SLSQP', 
                        constraints=constraints, tol=1e-29,)

            x = zip(coins, res.x)
            x = dict(filter(lambda x: x[-1]>0, x))
            coins = list(x.keys())
            if (sum(x.values())==1.):
                run = False
        results_weights = {'labels': list(x.keys()), 'data': [np.round(v * 100, 2) for v in x.values()]}

        asset = self.get_asset_shares(x, test)
        asset = asset.resample('W').mean().reset_index()
        asset ={'time': list(map(lambda x: pd.to_datetime(x).strftime('“%Y-%m-%d”'), asset.time.values.ravel())),
                'data': (100*asset.close.values.ravel()).tolist()}
        return results_weights, asset

