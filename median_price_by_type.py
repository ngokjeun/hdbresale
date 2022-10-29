import pandas as pd
import datetime as dt
df = pd.read_csv("resale_prices.csv")

df["Month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.month

print(df.groupby(["Month"]).count())
