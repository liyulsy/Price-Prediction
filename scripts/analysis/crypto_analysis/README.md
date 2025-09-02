# 加密货币数据分析工具

这个项目包含了一系列用于获取和分析加密货币数据的Python脚本。

## 文件结构

```
crypto_analysis/
├── data/              # 数据存储文件夹
│   └── 1D/           # 日线数据
├── scripts/          # Python脚本
│   ├── k_kline.py    # 数据获取脚本
│   ├── merge_data.py # 数据合并脚本
│   └── calculate_diff.py  # 计算价格变化率脚本
└── README.md         # 项目说明文件
```

## 使用说明

1. 获取数据：
```bash
python scripts/k_kline.py
```
- 获取8个币种的日线数据
- 数据保存在 data/1D/ 目录下

2. 合并数据：
```bash
python scripts/merge_data.py
```
- 合并所有币种的收盘价数据
- 生成 crypto_prices.csv 文件

3. 计算变化率：
```bash
python scripts/calculate_diff.py
```
- 计算每个币种的日涨跌幅（百分比）
- 生成 crypto_prices_rate.csv 文件

## 支持的币种

- AVAX-USDT
- BNB-USDT
- BTC-USDT
- DOGE-USDT
- ETH-USDT
- LTC-USDT
- SOL-USDT
- XRP-USDT 