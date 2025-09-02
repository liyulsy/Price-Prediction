import pandas as pd
import os
from datetime import datetime

def read_and_analyze_files(folder_path):
    stats = {}
    dataframes = {}
    min_length = float('inf')
    
    # 读取所有文件并分析
    for filename in os.listdir(folder_path):
        if filename.endswith('_1H.csv'):
            # 正确地从 'BTC_USDT_1H.csv' 中提取出 'BTC'
            base_name = filename.replace('_1H.csv', '')
            symbol = base_name.split('_')[0]
            file_path = os.path.join(folder_path, filename)
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            
            # 记录统计信息
            stats[symbol] = {
                'start_time': df['time'].min(),
                'end_time': df['time'].max(),
                'total_days': len(df)
            }
            
            # 保存数据框
            dataframes[symbol] = df
            
            # 更新最小数据量
            if len(df) < min_length:
                min_length = len(df)
    
    return stats, dataframes, min_length

def merge_and_align_data(dataframes, min_length):
    aligned_data = {}
    
    # 对每个币种的数据进行处理
    for symbol, df in dataframes.items():
        # 按时间排序并只取最新的min_length条数据
        df_sorted = df.sort_values('time', ascending=True)
        aligned_data[symbol] = df_sorted.tail(min_length)
    
    # 合并所有数据
    merged_df = None
    for symbol, df in aligned_data.items():
        # 只选择时间和收盘价
        selected_columns = df[['time', 'Close']]
        # 将列名改为 '币种-USDT' 格式，例如 'BTC-USDT'
        new_col_name = f"{symbol}-USDT"
        selected_columns = selected_columns.rename(columns={'Close': new_col_name})
        
        if merged_df is None:
            merged_df = selected_columns
        else:
            merged_df = pd.merge(merged_df, selected_columns, on='time', how='inner')
    
    # 将时间设置为索引
    merged_df.set_index('time', inplace=True)
    # 按时间降序排序
    merged_df.sort_index(inplace=True, ascending=False)
    
    return merged_df

def main():
    folder_path = 'crypto_analysis/data/raw_data/1H'
    output_file = 'crypto_analysis/data/processed_data/1H/all_1H.csv'
    
    # 读取和分析数据
    stats, dataframes, min_length = read_and_analyze_files(folder_path)
    
    # 打印统计信息
    print("\n数据统计信息:")
    print("-" * 50)
    for symbol, info in stats.items():
        print(f"\n{symbol}:")
        print(f"起始时间: {info['start_time']}")
        print(f"结束时间: {info['end_time']}")
        print(f"总天数: {info['total_days']}")
    
    print(f"\n最小数据量: {min_length} 天")
    
    # 合并数据
    merged_data = merge_and_align_data(dataframes, min_length)
    
    # 保存合并后的数据
    merged_data.to_csv(output_file)
    print(f"\n合并后的数据已保存到: {output_file}")
    print(f"合并后的数据维度: {merged_data.shape}")
    print("\n数据列名:")
    print(merged_data.columns.tolist())

if __name__ == "__main__":
    main() 