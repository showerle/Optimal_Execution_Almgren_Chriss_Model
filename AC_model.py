import random
import numpy as np
import pandas as pd
from constants import *
from benchmark_model import VWAMP


class AlmgrenModel(VWAMP):
    def __init__(self, trade_data, order_book, randomSeed=0, lambd=LLAMBDA):
        super().__init__(trade_data, LIQUIDATION_TIME, NUM_N, TOLERANCE, DIRECTION)

        # Set the random seed
        random.seed(randomSeed)

        # Below are parameters needed for every step calculation
        self.kappa = None
        self.kappa_hat = None
        self.eta_hat = None
        self.epsilon = None
        self.gamma = None
        self.eta = None
        self.singleStepVariance = None
        self.hourly_vola_ratio = None
        self.llambda = lambd
        self.prepare_order_data(order_book)

    def prepare_order_data(self, order):
        # calculate bid ask spread by order book data
        order = order[['timestamp', 'asks[0].price', 'bids[0].price']]
        order = order.set_index('timestamp')
        order.index = pd.to_datetime(order.index, unit='us')
        hourly_spread = order['asks[0].price'] - order['bids[0].price']
        self.hourly_spread = hourly_spread.resample(f'{self.tau}H').mean()

    def permanentImpact(self, sharesToSell):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        pi = self.gamma * sharesToSell
        return pi

    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        ti = (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)
        return ti

    def get_AC_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)
        st = self.epsilon * sharesToSell
        tt = self.eta_hat * (sharesToSell ** 2)
        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time)
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)
        fot = nft / dft
        return ft + st + (tt * fot)

    def get_AC_variance(self, sharesToSell):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.singleStepVariance * (sharesToSell ** 2)
        nst = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(self.kappa * (self.liquidation_time
                                                                                             - self.tau)) - self.liquidation_time * np.sinh(
            self.kappa * self.tau)
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)
        st = nst / dst
        return ft * st

    def compute_AC_utility(self, sharesToSell):
        # Calculate the AC Utility according to pg. 13 of the AC paper
        if self.liquidation_time == 0:
            return 0
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V

    def compute_trade_amount(self, time: int):
        # Calculate the trade amount for the optimal strategy according to equation (18) of the AC paper
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares
        st = np.cosh(self.kappa * (self.liquidation_time - (time + 1 - 0.5) * self.tau))
        trade_amount = st * ft

        return trade_amount

    def start_transactions(self, plot=True):
        # Set transactions on
        self.transacting = True

        # Set the initial capture to zero
        self.totalCapture = self.total_shares * self.startingPrice

        # Set the initial impacted price and price to the starting price
        self.prevImpactedPrice = self.startingPrice
        self.prevPrice = self.startingPrice

        # print(f'{self.__class__.__name__} algo transactions begins, initial total value {self.init_value}')

        # iterate transaction steps
        self.step()

        # print(f'{self.__class__.__name__} algo transactions end in {np.count_nonzero(self.trade_list != 0):} hour, '
        #       f'final total Capture {self.IS_list.sum() - self.init_value:.2f}')

        if plot:
            self.plot_trajectory()
            self.plot_is()


    def step(self):
        self.trade_list = np.zeros(self.num_n)
        self.IS_list = np.zeros(self.num_n)
        self.Utility_list = np.zeros(self.num_n)
        self.IS_diff_list = np.zeros(self.num_n)

        while self.transacting:
            for i in range(0, self.num_n):
                if self.timeHorizon == 0 or self.shares_remaining < self.tolerance:
                    # print('in here')
                    self.stop_transactions()
                    break

                trade_data = self.get_hourly_trade_data(i)
                hourly_trade_volume = trade_data['amount'].sum()
                self.hourly_vola_ratio = hourly_trade_volume / self.daily_volume  # Estimasted Hourly volatility using trade volume
                self.singleStepVariance = (
                                                      self.hourly_vola_ratio * self.startingPrice / 1000) ** 2  # Calculate single step variance
                self.basp = self.hourly_spread[i]
                self.epsilon = self.basp / 2  # Fixed Cost of Selling.
                self.eta = self.basp / (0.01 * hourly_trade_volume)  # Price Impact for Each 1% of Daily Volume Traded
                self.gamma = self.basp / (
                        0.1 * hourly_trade_volume)  # Permanent Impact for Each 10% of Daily Volume Traded
                self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
                self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
                self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau
                volumeToSell = min(self.compute_trade_amount(i), self.shares_remaining)
                volumeSold, singleSold, turnoverSold, traded_horizon = 0, 0, 0, 0

                while volumeSold < volumeToSell and traded_horizon < trade_data.shape[0]:
                    if (volumeToSell - volumeSold) < trade_data.iloc[traded_horizon,]['amount']:
                        singleSold = volumeToSell - volumeSold
                    elif (volumeToSell - volumeSold) < self.tolerance:
                        break
                    else:
                        singleSold = trade_data.iloc[traded_horizon,]['amount']
                    volumeSold += singleSold
                    turnoverSold += singleSold * trade_data.iloc[traded_horizon,]['price']
                    traded_horizon += 1

                self.trade_list[i] = volumeSold
                self.IS_list[i] = turnoverSold
                self.totalCapture += turnoverSold
                self.shares_remaining -= volumeSold

                # We don't add noise before the first trade
                if i == 0:
                    price = self.prevImpactedPrice
                else:
                    # Calculate the current stock price using arithmetic brownian motion
                    price = self.prevImpactedPrice + np.sqrt(
                        self.singleStepVariance * self.tau) * random.normalvariate(0, 1)

                    # Calculate the permanent and temporary impact on the stock price according the AC price dynamics
                    currentPermanentImpact = self.permanentImpact(volumeSold)
                    currentTemporaryImpact = self.temporaryImpact(volumeSold)

                    # Apply the temporary impact on the current stock price
                    estimated_exec_price = price - currentTemporaryImpact
                    estimated_shortfall = estimated_exec_price * volumeSold
                    estimated_shortfall_vs_real_shortfall = abs(estimated_shortfall - self.IS_list[i])
                    self.IS_diff_list[i] = estimated_shortfall_vs_real_shortfall
                    currentUtility = self.compute_AC_utility(self.shares_remaining)
                    self.Utility_list[i] = currentUtility

                    # Calculate the log return for the current step and save it in the logReturn deque
                    # self.logReturns.append(np.log(info.price / self.prevPrice))
                    # self.logReturns.popleft()

                    # Update the variables required for the next step
                    self.prevPrice = price
                    self.prevImpactedPrice = price - currentPermanentImpact
                    self.timeHorizon -= 1
