import collections
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.ticker as mticker
from constants import *


class VWAMP:
    def __init__(self, data: pd.DataFrame,
                 lqd_time=LIQUIDATION_TIME,
                 num_tr=NUM_N,
                 tolerance=TOLERANCE,
                 direction=DIRECTION):

        self.data = data
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.total_shares = TOTAL_SHARES
        self.startingPrice = STARTING_PRICE
        self.init_value = self.total_shares * self.startingPrice

        self.transacting = False
        self.shares_remaining = 0
        self.tolerance = tolerance
        self.tau = self.liquidation_time / self.num_n  # trading rate

        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n
        self.direction = direction
        self.prepare_data(self.data)

    def prepare_data(self, data):
        data = data[data['side'] == self.direction]
        data = data.set_index('timestamp')
        data.index = pd.to_datetime(data.index, unit='us')
        self.daily_volume = data['amount'].sum()

        # Group by transaction speed
        self.grouped_data = data.groupby(pd.Grouper(freq=f'{self.tau}H'))
        self.grouped_keys = [pd.Timestamp(group) for group, _ in self.grouped_data]

    def start_transactions(self, plot=True):
        self.transacting = True
        # print(f'{self.__class__.__name__} algo transactions begins, initial total value {self.init_value}')

        self.step()
        # print(f'{self.__class__.__name__} algo transactions end in {np.count_nonzero(self.trade_list != 0):} hour, '
        #       f'final total Capture {self.IS_list.sum() - self.init_value:.2f}')

        if plot:
            self.plot_trajectory()
            self.plot_is()

    def stop_transactions(self):
        self.transacting = False

    def get_hourly_trade_data(self, time: int):
        if len(self.grouped_keys) > self.num_n:
            return self.grouped_data.get_group(self.grouped_keys[time + 1])
        else:
            return self.grouped_data.get_group(self.grouped_keys[time])

    def step(self):
        self.trade_list = np.zeros(self.num_n)
        self.IS_list = np.zeros(self.num_n)

        while self.transacting:
            for i in range(0, self.num_n):
                if self.timeHorizon == 0 or self.shares_remaining < self.tolerance:
                    # print('in here')
                    self.stop_transactions()
                    break

                trade_data = self.get_hourly_trade_data(i)
                volumeToSell = min(trade_data['amount'].sum() / self.daily_volume * self.total_shares,
                                   self.shares_remaining)
                volumeSold, singleSold, turnoverSold,  traded_horizon = 0, 0, 0, 0

                # Use market order to trade
                while volumeSold < volumeToSell and traded_horizon < trade_data.shape[0]:
                    if (volumeToSell - volumeSold) < trade_data.iloc[traded_horizon, ]['amount']:
                        singleSold = volumeToSell - volumeSold
                    elif (volumeToSell - volumeSold) < self.tolerance:
                        break
                    else:
                        singleSold = trade_data.iloc[traded_horizon, ]['amount']

                    volumeSold += singleSold
                    turnoverSold += singleSold * trade_data.iloc[traded_horizon, ]['price']
                    traded_horizon += 1

                self.trade_list[i] = volumeSold
                self.IS_list[i] = turnoverSold
                self.shares_remaining -= volumeSold
                self.timeHorizon -= 1

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
        self.trade_list = np.zeros(self.num_n)
        self.IS_list = np.zeros(self.num_n)

        while self.transacting:
            for i in range(0, self.num_n):
                if self.timeHorizon == 0 or self.shares_remaining < self.tolerance:
                    # print('in here')
                    self.stop_transactions()
                    break

                trade_data = self.get_hourly_trade_data(i)
                # 按照时间交易
                volumeToSell = min(self.total_shares / self.num_n, self.shares_remaining)
                volumeSold, singleSold, turnoverSold, traded_horizon = 0, 0, 0, 0

                # 采用市价单成交
                while volumeSold < volumeToSell and traded_horizon < trade_data.shape[0]:
                    if (volumeToSell - volumeSold) < trade_data.iloc[traded_horizon, ]['amount']:
                        singleSold = volumeToSell - volumeSold
                    elif (volumeToSell - volumeSold) < self.tolerance:
                        break
                    else:
                        singleSold = trade_data.iloc[traded_horizon, ]['amount']
                    volumeSold += singleSold
                    turnoverSold += singleSold * trade_data.iloc[traded_horizon, ]['price']
                    traded_horizon += 1

                self.trade_list[i] = volumeSold
                self.IS_list[i] = turnoverSold
                self.shares_remaining -= volumeSold
                self.timeHorizon -= 1


