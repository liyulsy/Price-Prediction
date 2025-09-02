import pandas as pd
import os

# Define the directory containing the stock data
data_dir = "stock_analysis/stock_data"

# List of stock files and their corresponding ticker symbols
files_and_tickers = {
    "AAPL_Upward_Trend.csv": "AAPL",
    "NKLA_Downward_Trend.csv": "NKLA",
    "TSLA_Mixed_Trend.csv": "TSLA",
}

# An empty list to store individual stock dataframes
all_stocks = []

# Loop through the files
for filename, ticker in files_and_tickers.items():
    # Construct the full file path
    file_path = os.path.join(data_dir, filename)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Keep only the 'Date' and 'Close' columns
    df = df[['Date', 'Close']]
    
    # Set 'Date' as the index
    df = df.set_index('Date')
    
    # Rename the 'Close' column to the ticker symbol
    df = df.rename(columns={'Close': ticker})
    
    # Add the dataframe to our list
    all_stocks.append(df)

# Concatenate all dataframes in the list
merged_df = pd.concat(all_stocks, axis=1)

# Sort the dataframe by date in descending order
merged_df = merged_df.sort_index(ascending=False)

# Save the result to a new CSV file
output_filename = "merged_stocks.csv"
merged_df.to_csv(output_filename)

print(f"Successfully merged stock data into {output_filename}") 