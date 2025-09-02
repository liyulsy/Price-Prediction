import pandas as pd

# Define the input and output filenames
input_filename = "merged_stocks.csv"
output_filename = "stock_price_changes.csv"

# Read the merged stock data
# Set the first column ('Date') as the index
df = pd.read_csv(input_filename, index_col=0)

# The data is sorted in descending order by date.
# To calculate (price_today - price_yesterday) / price_yesterday,
# we can use pct_change with periods=-1, which looks at the next row.
price_change_df = df.pct_change(periods=-1)

# Rename the columns to indicate they are percentage changes
price_change_df.columns = [f'{col}_pct_change' for col in df.columns]

# Combine the original prices with the calculated changes
result_df = pd.concat([df, price_change_df], axis=1)

# The last row will have NaN for changes, so we can drop it if desired
result_df = result_df.dropna()

# Save the result to a new CSV file
result_df.to_csv(output_filename)

print(f"Successfully calculated price changes and saved to {output_filename}")
print("\nPreview of the result:")
print(result_df.head()) 