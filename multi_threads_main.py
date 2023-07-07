import pandas as pd
from AC_model import AlmgrenModel
from benchmark_model import VWAMP, TWAP
import warnings
from tqdm import tqdm
import threading

warnings.filterwarnings("ignore")

category = 'BTC'
# category = 'ETH'
trade_file_path = r"C:\Users\dell\data\trade"
order_file_path = r"C:\Users\dell\data\orderbook_25"

file_spectrum = 30


def run_vwap():
    vwap.start_transactions(plot=False)


def run_twap():
    twap.start_transactions(plot=False)


def run_acmodel():
    acmodel.start_transactions(plot=False)


if __name__ == '__main__':
    captureRes = pd.DataFrame(columns=['VWAP', 'TWAP', 'Almgren'])
    orderFillRes = pd.DataFrame(columns=['VWAP', 'TWAP', 'Almgren'])

    threads = []
    for i in tqdm(range(1, file_spectrum + 1)):
        trade_path = fr"{trade_file_path}\binance-futures_trades_2023-05-{i:02}_{category}USDT.csv\binance-futures_trades_2023-05-{i:02}_{category}USDT.csv"
        trade_data = pd.read_csv(trade_path)

        order_path = fr"{order_file_path}\binance-futures_book_snapshot_25_2023-05-{i:02}_{category}USDT.csv\binance-futures_book_snapshot_25_2023-05-{i:02}_{category}USDT.csv"
        order_book = pd.read_csv(order_path)

        vwap = VWAMP(trade_data)
        twap = TWAP(trade_data)
        acmodel = AlmgrenModel(trade_data, order_book)

        vwap_thread = threading.Thread(target=run_vwap)
        twap_thread = threading.Thread(target=run_twap)
        acmodel_thread = threading.Thread(target=run_acmodel)

        threads.append(vwap_thread)
        threads.append(twap_thread)
        threads.append(acmodel_thread)

        vwap_thread.start()
        twap_thread.start()
        acmodel_thread.start()

        for thread in threads:
            thread.join()

        captureRes = captureRes.append({'VWAP': f"{vwap.IS_list.sum() - vwap.init_value:.2f}",
                                        'TWAP': f"{twap.IS_list.sum() - twap.init_value:.2f}",
                                        'Almgren': f"{acmodel.IS_list.sum() - acmodel.init_value:.2f}"
                                        }, ignore_index=True)

        orderFillRes = orderFillRes.append({'VWAP': f"{vwap.trade_list.sum() / vwap.total_shares * 100:.2f}%",
                                            'TWAP': f"{twap.trade_list.sum() / twap.total_shares * 100:.2f}%",
                                            'Almgren': f"{acmodel.trade_list.sum() / acmodel.total_shares * 100:.2f}%"
                                            }, ignore_index=True)

    captureRes.to_csv(f"datasets/captureResults_{category}.csv")
    orderFillRes.to_csv(f"datasets/orderFillResults_{category}.csv")

    print(captureRes)
    print(orderFillRes)

    is_stats = captureRes
    is_stats['vwap_VS_almgren'] = (is_stats.VWAP - is_stats.Almgren) / is_stats.VWAP
    is_stats['vwap_VS_almgren'].plot()
