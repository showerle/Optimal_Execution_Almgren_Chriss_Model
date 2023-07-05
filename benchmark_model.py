import collections
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.ticker as mticker

TOTAL_SHARES = 1000  # Total number of shares to sell
LIQUIDATION_TIME = 24  # How many hours to sell all the shares.
NUM_N = 24  # Number of trades
TOLERANCE = 0.01  # Minimum number of stocks one can sell(in BTCUSDT is 0.01 USDT)
DIRECTION = 'buy'


class VWAMP:
    def __init__(self, data: pd.DataFrame,
                 lqd_time=LIQUIDATION_TIME,
                 num_tr=NUM_N,
                 all_share=TOTAL_SHARES,
                 tolerance=TOLERANCE,
                 direction=DIRECTION):
        self.direction = direction
        self.data = self.prepare_data(data)

        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.total_shares = all_share
        self.startingPrice = self.data['price'].iloc[0, ]

        self.transacting = False
        self.shares_remaining = 0
        self.tolerance = tolerance
        self.tau = self.liquidation_time / self.num_n  # trading rate

        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n

    def prepare_data(self, data):
        data = data[data['side'] == self.direction]
        data = data.set_index('timestamp')
        data.index = pd.to_datetime(data.index, unit='us')
        return data

    def start_transactions(self):
        # Set transactions on
        self.transacting = True
        self.init_value = self.total_shares * self.startingPrice
        print(f'{self.__class__.__name__} algo transactions begins, initial total value {self.init_value}')

        self.trade_list, self.IS_list = self.step()
        print(f'{self.__class__.__name__} algo transactions end in {np.count_nonzero(self.trade_list != 0):} hour, '
              f'final total Capture {self.IS_list.sum() - self.init_value:.2f}')
        self.plot_trajectory()
        self.plot_is()

    def stop_transactions(self):
        # Stop transacting
        self.transacting = False

    def step(self):
        trade_list = np.zeros(self.num_n)
        IS_list = np.zeros(self.num_n)

        # 按照交易速度进行分组
        grouped_data = self.data.groupby(pd.Grouper(freq=f'{self.tau}H'))
        grouped_keys = [pd.Timestamp(group) for group, _ in grouped_data]

        while self.transacting:
            for i in tqdm(range(0, self.num_n)):
                if self.timeHorizon == 0 or self.shares_remaining < self.tolerance:
                    # print('in here')
                    self.stop_transactions()
                    break

                trade_data = grouped_data.get_group(grouped_keys[i])
                volumeToSell = trade_data['amount'].sum() / self.data['amount'].sum() * self.total_shares
                traded_horizon = 0

                # 采用市价单成交
                while volumeToSell > self.tolerance and traded_horizon < trade_data.shape[0]:
                    volumeToSell -= trade_data.iloc[traded_horizon, ]['amount']
                    traded_horizon += 1

                # 计算出每个交易区间的VWAP
                volumeSelled = trade_data.iloc[:traded_horizon, ]['amount'].sum()
                trade_list[i] = volumeSelled

                IS_list[i] = (trade_data.iloc[:traded_horizon, ]['amount'] * trade_data.iloc[:traded_horizon, ][
                    'price']).sum()
                self.shares_remaining -= volumeSelled
                self.timeHorizon -= 1

        return trade_list, IS_list

    def plot_trajectory(self):
        new_trl = np.insert(self.trade_list, 0, 0)
        df = pd.DataFrame(data=list(range(self.num_n + 1)), columns=['Trade Number'], dtype='float64')
        df['Stocks Sold'] = new_trl
        df['Stocks Remaining'] = (np.ones(self.num_n + 1) * self.total_shares) - np.cumsum(new_trl)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        df.iloc[1:].plot.scatter(x='Trade Number', y='Stocks Sold', c='Stocks Sold', colormap='gist_rainbow',
                                 alpha=1, sharex=False, s=50, colorbar=False, ax=axes[0])

        # Plot a line through the points of the scatter plot of the trade list
        axes[0].plot(df['Trade Number'].iloc[1:], df['Stocks Sold'].iloc[1:], linewidth=2.0, alpha=0.5)
        axes[0].set_facecolor(color='k')
        yNumFmt = mticker.StrMethodFormatter('{x:,.0f}')
        axes[0].yaxis.set_major_formatter(yNumFmt)
        axes[0].set_title(f'{self.__class__.__name__} algo Trading List')

        # Make a scatter plot of the number of stocks remaining after each trade
        df.plot.scatter(x='Trade Number', y='Stocks Remaining', c='Stocks Remaining', colormap='gist_rainbow',
                        alpha=1, sharex=False, s=50, colorbar=False, ax=axes[1])

        # Plot a line through the points of the scatter plot of the number of stocks remaining after each trade
        axes[1].plot(df['Trade Number'], df['Stocks Remaining'], linewidth=2.0, alpha=0.5)
        axes[1].set_facecolor(color='k')
        yNumFmt = mticker.StrMethodFormatter('{x:,.0f}')
        axes[1].yaxis.set_major_formatter(yNumFmt)
        axes[1].set_title(f' {self.__class__.__name__} algo Trading Trajectory')

        # Set the spacing between plots
        plt.subplots_adjust(wspace=0.4)
        plt.show()

        print(f'{self.__class__.__name__} algo Number of Shares Sold: {new_trl.sum():.2f}')

    def plot_is(self):

        shortfall_hist = self.IS_list
        print(f'\n{self.__class__.__name__} algo Average Implementation Shortfall: {shortfall_hist.mean():.2f}')
        print(f'{self.__class__.__name__} Standard Deviation of the Implementation Shortfall: {shortfall_hist.std():.2f}')

        plt.plot(shortfall_hist, 'cyan', label='')
        plt.xlim(0, self.num_n)
        ax = plt.gca()
        ax.set_facecolor('k')
        ax.set_xlabel('Episode', fontsize=15)
        ax.set_ylabel(f'{self.__class__.__name__} Implementation Shortfall', fontsize=15)
        ax.axhline(shortfall_hist.mean(), 0, 1, color='m', label='Average')
        yNumFmt = mticker.StrMethodFormatter('${x:,.0f}')
        ax.yaxis.set_major_formatter(yNumFmt)
        plt.legend()
        plt.show()


class TWAP(VWAMP):
    def step(self):
        trade_list = np.zeros(self.num_n)
        IS_list = np.zeros(self.num_n)

        # 按照交易速度进行分组
        grouped_data = self.data.groupby(pd.Grouper(freq=f'{self.tau}H'))
        grouped_keys = [pd.Timestamp(group) for group, _ in grouped_data]

        while self.transacting:
            for i in tqdm(range(0, self.num_n)):
                if self.timeHorizon == 0 or self.shares_remaining < self.tolerance:
                    # print('in here')
                    self.stop_transactions()
                    break

                trade_data = grouped_data.get_group(grouped_keys[i])
                # 按照时间交易
                volumeToSell = self.total_shares / self.num_n
                traded_horizon = 0

                # 采用市价单成交
                while volumeToSell > self.tolerance and traded_horizon < trade_data.shape[0]:
                    volumeToSell -= trade_data.iloc[traded_horizon, ]['amount']
                    traded_horizon += 1

                volumeSelled = trade_data.iloc[:traded_horizon, ]['amount'].sum()
                trade_list[i] = volumeSelled
                # self.totalCapture += (trade_data.iloc[:traded_horizon, ]['amount'] * trade_data.iloc[:traded_horizon, ][
                #     'price']).sum()
                IS_list[i] = IS_list[i] = (trade_data.iloc[:traded_horizon, ]['amount'] * trade_data.iloc[:traded_horizon, ][
                    'price']).sum()
                self.shares_remaining -= volumeSelled
                self.timeHorizon -= 1

        return trade_list, IS_list




def main():
    data = pd.read_csv(fr"C:\Users\dell\data\trade\binance-futures_trades_2023-05-01_BTCUSDT.csv\binance-futures_trades_2023-05-01_BTCUSDT.csv")
    vwap = VWAMP(data)
    vwap.start_transactions()

    twap = TWAP(data)
    twap.start_transactions()


main()
