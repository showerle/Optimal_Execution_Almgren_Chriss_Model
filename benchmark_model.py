import collections
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.ticker as mticker


TOTAL_SHARES = 100  # Total number of shares to sell
LIQUIDATION_TIME = 30  # How many days to sell all the shares.
NUM_N = 30  # Number of trades
POV = 0.001  # PARTICIPATION OF VOLUME
TOLERANCE = 0.01   # Minimum number of stocks one can sell(in BTCUSDT is 0.01 USDT)


class VWAMP:
    def __init__(self, data: pd.DataFrame,
                 lqd_time=LIQUIDATION_TIME,
                 num_tr=NUM_N,
                 pov=POV,
                 all_share=TOTAL_SHARES,
                 tolerance=TOLERANCE):
        self.data = data
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.pov = pov / 100
        self.total_shares = all_share
        self.startingPrice = self.data['price'].loc[0, ]

        self.transacting = False
        self.shares_remaining = 0
        self.tolerance = tolerance

        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n

    def start_transactions(self):
        # Set transactions on
        self.transacting = True
        self.init_value = self.total_shares * self.startingPrice
        self.totalCapture = self.init_value

        trade_list, IS_list = self.step()
        print(trade_list)
        self.plot(trade_list)
        print(f'transactions begins,initial total Capture {self.totalCapture}')
        print(f'transactions begins,initial total Capture {self.totalCapture}')


    def stop_transactions(self):
        # Stop transacting
        self.transacting = False
        print(f'transactions end, final total Capture {self.totalCapture}')

    def step(self):
        trade_list = np.zeros(self.num_n)
        IS_list = np.zeros(self.num_n)

        while self.transacting:
            for i in tqdm(range(0, self.num_n)):
                if self.timeHorizon == 0 or abs(self.shares_remaining) < self.tolerance:
                    # print('in here')
                    self.stop_transactions()
                    break
                volumeToSell = self.data.loc[i, 'volume'] * self.pov
                exec_price = self.data.loc[i, 'price']
                trade_list[i] = volumeToSell
                self.totalCapture += volumeToSell * exec_price
                IS_list[i] = self.totalCapture - self.init_value

                self.shares_remaining -= volumeToSell
                self.timeHorizon -= 1

        return trade_list, IS_list

    def plot(self, data: np.ndarray):
        new_trl = np.insert(data, 0, 0)
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
        axes[0].set_title('Trading List')

        # Make a scatter plot of the number of stocks remaining after each trade
        df.plot.scatter(x='Trade Number', y='Stocks Remaining', c='Stocks Remaining', colormap='gist_rainbow',
                        alpha=1, sharex=False, s=50, colorbar=False, ax=axes[1])

        # Plot a line through the points of the scatter plot of the number of stocks remaining after each trade
        axes[1].plot(df['Trade Number'], df['Stocks Remaining'], linewidth=2.0, alpha=0.5)
        axes[1].set_facecolor(color='k')
        yNumFmt = mticker.StrMethodFormatter('{x:,.0f}')
        axes[1].yaxis.set_major_formatter(yNumFmt)
        axes[1].set_title('Trading Trajectory')

        # Set the spacing between plots
        plt.subplots_adjust(wspace=0.4)
        plt.show()

        print('\nNumber of Shares Sold: {:,.0f}\n'.format(new_trl.sum()))


def main():
    data = pd.read_csv(f"datasets/trade/BTCUSDT_June.csv")
    vwap = VWAMP(data)
    vwap.start_transactions()


main()
