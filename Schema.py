import pandas as pd

df = pd.read_csv(r'/Users/kenny/Documents/Documents - Kehinde’s MacBook Pro (2)/Datasets/Retail_Inventory_Forecasting_Project/Data/Raw/retail_store_inventory.csv')
print(df.info())
print(df.head(3).to_string())
