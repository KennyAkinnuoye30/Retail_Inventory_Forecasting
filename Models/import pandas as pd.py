import pandas as pd
import sqlite3
import os

# === Load and Prepare Data (Task 1) ===
folder = r'/Users/kenny/Documents/simpletire'
conn = sqlite3.connect('simpletire.db')

orders = pd.read_csv(os.path.join(folder, 'ORDERS.csv'))
customers = pd.read_csv(os.path.join(folder, 'CUSTOMERS.csv'))
products = pd.read_csv(os.path.join(folder, 'PRODUCTS.csv'))
order_items = pd.read_csv(os.path.join(folder, 'ORDER_ITEM_DETAIL.csv'))

orders.to_sql('ORDERS', conn, index=False, if_exists='replace')
customers.to_sql('CUSTOMERS', conn, index=False, if_exists='replace')
products.to_sql('PRODUCTS', conn, index=False, if_exists='replace')
order_items.to_sql('ORDER_ITEM_DETAIL', conn, index=False, if_exists='replace')

print("✅ All CSVs successfully loaded into simpletire.db")

# === Fix PRODUCT_IDs (Normalize IDs like P101 → P1001) ===
products['PRODUCT_ID'] = products['PRODUCT_ID'].str.replace(
    r'^P1(\d{2})$', r'P10\1', regex=True
)
products.to_sql('PRODUCTS', conn, index=False, if_exists='replace')
print("✅ PRODUCTS IDs normalized to P1001–P1015 and table replaced in SQLite.")


sql =  """
SELECT
c.NAME AS customer_name,
oi.PRODUCT_ID,
p.BRAND,
ROUND(SUM(oi.QUANTITY * oi.UNIT_PRICE), 2) AS total_sales
FROM ORDERS o
JOIN CUSTOMERS c ON c.CUSTOMER_ID = o.CUSTOMER_ID
JOIN ORDER_ITEM_DETAIL oi ON oi.ORDER_ID = o.ORDER_ID
JOIN PRODUCTS p ON p.PRODUCT_ID = oi.PRODUCT_ID
GROUP BY c.NAME, oi.PRODUCT_ID, p.BRAND
ORDER BY c.NAME, total_sales DESC;

"""
df = pd.read_sql_query(sql, conn)
print(df)

'''
# === Scenario 1: Data Quality Validation ===

# Task 2 – Calculate line subtotal
order_items['LINE_SUBTOTAL'] = order_items['QUANTITY'] * order_items['UNIT_PRICE']

# Task 3 – Aggregate totals per order
calc = (
    order_items
    .groupby('ORDER_ID', as_index=False)['LINE_SUBTOTAL']
    .sum()
    .rename(columns={'LINE_SUBTOTAL': 'CALCULATED_SALES'})
)

# Task 4 – Merge aggregated results onto ORDERS
check = orders.merge(calc, on='ORDER_ID', how='left')
check['CALCULATED_SALES'] = check['CALCULATED_SALES'].fillna(0.0)

# Task 5 – Create SALES_CHECK column (difference)
check['SALES_CHECK'] = check['TOTAL_SALES'] - check['CALCULATED_SALES']

# Task 6 – Identify potential issues
issues = check[check['SALES_CHECK'].abs() > 0.01]

print("=== Data Quality Check ===")
print(check[['ORDER_ID','TOTAL_SALES','CALCULATED_SALES','SALES_CHECK']])

print("\n=== Potential Issues (> $0.01 diff) ===")
print(issues[['ORDER_ID','TOTAL_SALES','CALCULATED_SALES','SALES_CHECK']])

# Result summary
print(
    "\nResult: No discrepancies detected within ±$0.01."
    if issues.empty
    else "\nResult: Discrepancies found; see rows above."
)

# === Scenario 2: Brand performance by year ===

# 1–2) Create one comprehensive DataFrame (ORDERS → ORDER_ITEM_DETAIL → PRODUCTS)
df = (orders
      .merge(order_items, on='ORDER_ID', how='inner')
      .merge(products,   on='PRODUCT_ID', how='inner'))

# 3–4) Ensure date is datetime and extract year
df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'])
df['ORDER_YEAR'] = df['ORDER_DATE'].dt.year

# 5) Line total = Quantity × Unit Price
df['LINE_TOTAL'] = df['QUANTITY'] * df['UNIT_PRICE']

# 6–8) Group by year & brand → total annual sales; sort by year then sales desc
report = (df.groupby(['ORDER_YEAR', 'BRAND'], as_index=False)['LINE_TOTAL']
            .sum()
            .rename(columns={'LINE_TOTAL': 'TOTAL_ANNUAL_SALES'})
            .sort_values(['ORDER_YEAR', 'TOTAL_ANNUAL_SALES'], ascending=[True, False]))

print("=== Scenario 2: Brand performance by year ===")
print(report)

'''


conn.close()



'''
sql = """
SELECT
  DATE(o.ORDER_DATE)                  AS order_date,
  COUNT(DISTINCT o.ORDER_ID)          AS total_orders,
  SUM(oi.QUANTITY)                    AS total_quantity,
  COUNT(DISTINCT o.CUSTOMER_ID)       AS total_customers
FROM ORDERS AS o
JOIN ORDER_ITEM_DETAIL AS oi
  ON oi.ORDER_ID = o.ORDER_ID
GROUP BY DATE(o.ORDER_DATE)
ORDER BY order_date;

"""



'''



