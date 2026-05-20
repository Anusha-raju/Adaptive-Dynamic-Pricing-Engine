# env.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class PricingBanditEnv:
    def __init__(self, xlsx_path="online_retail_II.xlsx", K=5):
        self.xlsx_path = xlsx_path
        self.K = K
        self.data = None
        self.T = None
        self.arm_means = None

        self._preprocess()
        self._stationarity_report()

    def _load_excel(self):
        sheets = pd.read_excel(self.xlsx_path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
        df.columns = [c.strip() for c in df.columns]
        return df

    def _preprocess(self):
        df = self._load_excel()

        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["date"] = df["InvoiceDate"].dt.date
        df["revenue"] = df["Quantity"] * df["Price"]

        # Choose top country
        top_country = df.groupby("Country")["Quantity"].sum().idxmax()
        df = df[df["Country"] == top_country]

        # Choose top selling SKU
        top_sku = df.groupby("StockCode")["revenue"].sum().idxmax()
        df_sku = df[df["StockCode"] == top_sku].copy()
        print(f"[ENV] Using Country = {top_country}, StockCode = {top_sku}")

        daily = (
            df_sku.groupby("date")
                  .agg(avg_price=("Price", "mean"),
                       qty=("Quantity", "sum"),
                       revenue=("revenue", "sum"))
                  .reset_index()
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        # Price buckets → arms
        try:
            daily["bucket"] = pd.qcut(
                daily["avg_price"], q=self.K, labels=False, duplicates="drop"
            ).astype(int)
        except:
            daily["bucket"] = pd.cut(
                daily["avg_price"], bins=self.K, labels=False
            ).astype(int)

        self.K = daily["bucket"].nunique()

        bucket_avg = daily.groupby("bucket")["revenue"].mean().to_dict()
        self.arm_means = np.array([bucket_avg[k] for k in range(self.K)])

        T = len(daily)
        data = np.zeros((T, self.K))

        for t in range(T):
            b = daily.loc[t, "bucket"]
            r = daily.loc[t, "revenue"]
            for k in range(self.K):
                data[t, k] = r if k == b else bucket_avg[k]

        self.data = data
        self.T = T
        print(f"[ENV] Loaded. data shape = {self.data.shape}")

    def _stationarity_report(self):
        print("\n[ENV] ADF Stationarity Test per Arm:")
        for k in range(self.K):
            stat, p = adfuller(self.data[:, k])[:2]
            print(f"  Arm {k}: ADF stat = {stat:.3f}, p = {p:.3f}")
