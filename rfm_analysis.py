import pandas as pd
import numpy as np
from datetime import datetime

# 1. GENERATE MOCK TRANSACTION DATA
# Simulating a database of customer purchases in 2026
data = {
    'transaction_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'customer_id':   [501,  502,  501,  503,  504,  502,  501,  505,  503,  502],
    'purchase_date': [
        '2026-01-15', '2026-02-10', '2026-03-22', '2026-05-01', '2026-06-01',
        '2026-06-10', '2026-06-15', '2026-01-20', '2026-06-14', '2026-06-16'
    ],
    'amount_spent':  [120.00, 45.00, 200.00, 350.00, 15.00, 60.00, 80.00, 500.00, 150.00, 75.00]
}

df = pd.DataFrame(data)
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Set "today's date" for recency calculation (Simulated as June 17, 2026)
today_date = datetime(2026, 6, 17)

print("--- RAW TRANSACTION DATA ---")
print(df.head(3))
print("\n" + "="*50 + "\n")

# 2. CALCULATE RFM METRICS PER CUSTOMER
rfm = df.groupby('customer_id').agg({
    'purchase_date': lambda x: (today_date - x.max()).days, # Recency: Days since last order
    'transaction_id': 'count',                             # Frequency: Total orders
    'amount_spent': 'sum'                                  # Monetary: Total revenue
}).rename(columns={
    'purchase_date': 'Recency',
    'transaction_id': 'Frequency',
    'amount_spent': 'Monetary'
})

print("--- CALCULATED RFM VALUES ---")
print(rfm)
print("\n" + "="*50 + "\n")

# 3. ASSIGN QUANTILE SCORES (1 to 3 for scale simplicity in small data)
# pd.qcut assigns scores based on percentiles. 
# For Recency, smaller days away is better, so labels are [3, 2, 1]
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=3, labels=[3, 2, 1], duplicates='drop')
rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=3, labels=[1, 2, 3], duplicates='drop')
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=3, labels=[1, 2, 3], duplicates='drop')

# 4. SEGMENT CUSTOMERS BASED ON LOGIC
def define_segment(row):
    # High frequency, high monetary, low recency days
    if row['R_Score'] == 3 and row['M_Score'] == 3:
        return 'Champions (Core Devoted)'
    # High recency days (haven't bought in a long time)
    elif row['R_Score'] == 1:
        return 'At Risk / Churning'
    # Medium active metrics
    else:
        return 'Regular Active Customer'

rfm['Customer_Segment'] = rfm.apply(define_segment, axis=1)

print("--- FINAL PORTFOLIO SEGMENATION MATRIX ---")
print(rfm[['Recency', 'Frequency', 'Monetary', 'Customer_Segment']])

