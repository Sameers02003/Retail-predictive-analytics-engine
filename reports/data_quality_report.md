# Data quality report

Generated: 2026-10-26 15:59

## Rows per table

- stores: 12 rows
- products: 300 rows
- customers: 4,015 rows
- orders: 82,089 rows
- order_items: 204,230 rows
- transactions: 84,390 rows

## Issues found

- **[Low] customers.city**: blank value: 50 rows (1.25%). Action: Fill with 'Unknown'; city is not a model input
- **[Medium] orders.customer_id**: blank value: 246 rows (0.3%). Action: Keep for sales forecasting; exclude from churn (can't be linked to a customer)
- **[High] orders.order_ts**: blank value: 7 rows (0.01%). Action: Recover date from the successful payment time
- **[Medium] order_items.unit_price**: blank value: 40 rows (0.02%). Action: Fill with the product's list price
- **[Low] customers.customer_id**: exact duplicate row: 15 rows (0.37%). Action: Drop the extra copies
- **[High] orders.order_id**: exact duplicate row: 326 rows (0.4%). Action: Drop the extra copies (otherwise sales are double counted)
- **[High] orders.order_id**: same ID, different content: 1 rows (0.0%). Action: Keep the copy that has a customer_id
- **[High] orders.order_ts**: unparseable date: 8 rows (0.01%). Action: Recover date from the successful payment time
- **[High] orders.order_ts**: date in the future: 10 rows (0.01%). Action: Recover date from the successful payment time
- **[High] order_items.quantity**: below allowed minimum: 60 rows (0.03%). Action: Remove these item lines (impossible purchases)
- **[Medium] products.category**: value not in allowed list: 36 rows (12.0%). Action: Map variants to the 5 standard categories; assume 'Apparel' = 'Fashion'

## Checks that passed

- All 6 files have the expected columns (schema check, Step 1)
- All discounts are between 0 and 100%
- All dates in transactions, customers and stores are valid
- All IDs link to an existing parent record (5 links checked)
- Every order has at least one item and a successful payment

## Noted, not treated as errors

- Some quantities (up to 117) and payment amounts (up to Rs 14.4 lakh) are far above normal. They break no rule, so they are not treated as data errors; they are investigated in Step 8 (anomaly detection).