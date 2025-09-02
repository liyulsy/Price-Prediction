import pandas as pd

def calculate_price_change_rate():
    # 读取原始价格数据
    prices_df = pd.read_csv('crypto_prices.csv')
    
    # 将时间列设置为索引
    prices_df['time'] = pd.to_datetime(prices_df['time'])
    prices_df.set_index('time', inplace=True)
    
    # 按时间升序排列，以便正确计算变化率
    prices_df = prices_df.sort_index(ascending=True)
    
    # 计算价格变化率（百分比）
    # pct_change() 计算公式：(当前价格 - 前一天价格) / 前一天价格 * 100
    rate_df = prices_df.pct_change() * 100
    
    # 删除第一行（因为它是NaN）
    rate_df = rate_df.iloc[1:]
    
    # 按时间倒序排列
    rate_df = rate_df.sort_index(ascending=False)
    
    # 保存价格变化率数据
    output_file = 'crypto_prices_rate.csv'
    rate_df.to_csv(output_file)
    
    print("\n数据处理完成:")
    print(f"价格变化率数据已保存到: {output_file}")
    print(f"数据维度: {rate_df.shape}")
    print("\n数据列名:")
    print(rate_df.columns.tolist())
    print("\n价格变化率数据前5行样例（单位：%）:")
    print(rate_df.head().round(2))  # 四舍五入到2位小数

if __name__ == "__main__":
    calculate_price_change_rate() 