import pandas as pd
from tqdm import tqdm
from AC_model import AlmgrenModel
from benchmark_model import VWAMP, TWAP
import warnings

warnings.filterwarnings("ignore")

category = 'BTC'
# category = 'ETH'
trade_file_path = r"C:\Users\dell\data\trade"
order_file_path = r"C:\Users\dell\data\orderbook_25"

file_spectrum = 30

if __name__ == '__main__':
    captureRes = pd.DataFrame(columns=['VWAP', 'TWAP', 'Almgren'])
    orderFillRes = pd.DataFrame(columns=['VWAP', 'TWAP', 'Almgren'])

    for i in tqdm(range(1, file_spectrum + 1)):
        trade_path = fr"{trade_file_path}\binance-futures_trades_2023-05-{i:02}_{category}USDT.csv\binance-futures_trades_2023-05-{i:02}_{category}USDT.csv"
        trade_data = pd.read_csv(trade_path)

        order_path = fr"{order_file_path}\binance-futures_book_snapshot_25_2023-05-{i:02}_{category}USDT.csv\binance-futures_book_snapshot_25_2023-05-{i:02}_{category}USDT.csv"
        order_book = pd.read_csv(order_path)

        vwap = VWAMP(trade_data)
        twap = TWAP(trade_data)
        acmodel = AlmgrenModel(trade_data, order_book)

        vwap.start_transactions(plot=False)
        twap.start_transactions(plot=False)
        acmodel.start_transactions(plot=False)

        captureRes = captureRes.append({'VWAP': f"{vwap.IS_list.sum() - vwap.init_value:.2f}",
                                        'TWAP': f"{twap.IS_list.sum() - twap.init_value:.2f}",
                                        'Almgren': f"{acmodel.IS_list.sum() - acmodel.init_value:.2f}"
                                        }, ignore_index=True)
        orderFillRes = orderFillRes.append({'VWAP': f"{vwap.trade_list.sum() / vwap.total_shares * 100:.2f}%",
                                            'TWAP': f"{twap.trade_list.sum() / twap.total_shares * 100:.2f}%",
                                            'Almgren': f"{acmodel.trade_list.sum() / acmodel.total_shares * 100:.2f}%"
                                            }, ignore_index=True)

    print(captureRes)
    print(orderFillRes)
