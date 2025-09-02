import requests
import csv
import time
import urllib3
import os
from time_transfer import unix_timestamp_to_beijing_time

urllib3.disable_warnings()

# 创建保存数据的文件夹
DATA_FOLDER = '1D'
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# List of cryptocurrency symbols to fetch
CRYPTO_SYMBOLS = [
    'AVAX-USDT',
    'BNB-USDT',
    'BTC-USDT',
    'DOGE-USDT',
    'ETH-USDT',
    'LTC-USDT',
    'SOL-USDT',
    'XRP-USDT'
]

def initialize_csv_file(file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'Open', 'High', 'Low', 'Close', 'vol'])

def ouyi_symbols(symbol, after):
    output_list = []
    try:
        # 构造请求 URL
        url = f'https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar=1D&after={after}&before=1527696000000'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
        }
        
        # 发起请求
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        datas = response.json()

        # 检查返回数据结构
        if "data" in datas:
            data_list = datas["data"]
            for item in data_list:
                ts = item[0]  # 时间戳
                beijing_time = unix_timestamp_to_beijing_time(int(ts))
                o = item[1]   # 开盘价
                h = item[2]   # 最高价
                l = item[3]   # 最低价
                c = item[4]   # 收盘价
                vol = item[7] # 成交量
                output_list.append({
                    'time': beijing_time,
                    'open_price': o,
                    'highest_price': h,
                    'lowest_price': l,
                    'close_price': c,
                    'vol': vol
                })
            
            # 获取最后一条时间戳
            last_ts = data_list[-1][0] if data_list else None
            time.sleep(2)
        else:
            print(f"No data found in response for {symbol}")
            last_ts = None

    except requests.exceptions.RequestException as e:
        print(f"Request error for {symbol}: {e}")
        last_ts = None
    except ValueError as ve:
        print(f"JSON parsing error for {symbol}: {ve}")
        last_ts = None

    return output_list, last_ts

def fetch_data_with_retry(symbol, initial_timestamp, retry_limit=10):
    filename = os.path.join(DATA_FOLDER, f"{symbol.replace('-', '_')}_1D.csv")
    current_timestamp = initial_timestamp
    all_data = []
    retry_count = 0
    initialize_csv_file(filename)

    while current_timestamp and retry_count < retry_limit:
        print(f"Fetching {symbol} data after timestamp: {current_timestamp}")
        data, last_ts = ouyi_symbols(symbol, current_timestamp)
        
        if data:
            all_data.extend(data)
            current_timestamp = last_ts

            # # 检查时间戳差值是否在一分钟以内，若是则结束循环
            # if abs(int(current_timestamp) - 1609430400000) <= 1 * 1000:
            #     print("Time difference is within 1 minute, ending loop.")
            #     break

            retry_count = 0  # Reset retry count on success
        else:
            print(f"Error occurred for {symbol}. Retrying...")
            retry_count += 1
            time.sleep(2)  # 等待2秒再重试

    if retry_count == retry_limit:
        print(f"Reached retry limit for {symbol}. Exiting.")

    # 保存数据到文件
    if all_data:
        save_data_to_csv(all_data, filename)
        print(f"Data saved to {filename}")    
    
    return all_data

def save_data_to_csv(data, filename):
    # 定义字段名称
    fieldnames = ['time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'vol']
    
    # 打开文件并写入数据
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 如果是第一次写入文件，写入表头
        if file.tell() == 0:
            writer.writeheader()
        
        # 写入每一条数据
        writer.writerows(data)

# 主程序执行
def main():
    start_timestamp = "1748620800000"  # 起始时间戳 2025-06-01 01:00:00
    
    print(f"数据将保存到 {os.path.abspath(DATA_FOLDER)} 文件夹中")
    
    for symbol in CRYPTO_SYMBOLS:
        print(f"\n开始获取 {symbol} 的数据")
        retrieved_data = fetch_data_with_retry(symbol, start_timestamp)
        print(f"{symbol} 总共获取到 {len(retrieved_data)} 条记录")
        time.sleep(5)  # Add delay between different symbols to avoid rate limiting

if __name__ == "__main__":
    main()
