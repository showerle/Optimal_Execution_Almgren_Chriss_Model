from tardis_dev import datasets

datasets.download(
    exchange="binance-futures",
    data_types=[
            "trades"
    ],
    from_date="2023-05-01",
    to_date="2023-06-01",
    symbols=["BTCUSDT", "ETHUSDT"],
    api_key="TD.ch1avFy1nY2ZWp2K.UuERYMSLcaQRL-l.0OWfFqzNcO4cSe6.wtEg0Hka6TybKqi.CntpAvoUKjH0Eyj.BZ-B",
)

