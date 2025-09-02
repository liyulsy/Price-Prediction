import json
import os
import pandas as pd
from datetime import datetime

def check_news_time_span(features_dir):
    """
    检查指定目录下所有币种新闻特征文件的时间跨度。

    Args:
        features_dir (str): 包含 _features.json 文件的目录路径。
    """
    print(f"正在检查目录: {features_dir}")
    results = {}

    try:
        all_files = [f for f in os.listdir(features_dir) if f.endswith('_features.json')]
    except FileNotFoundError:
        print(f"错误: 找不到目录 {features_dir}")
        return

    if not all_files:
        print("错误: 目录下没有找到 *_features.json 文件。")
        return

    for filename in all_files:
        file_path = os.path.join(features_dir, filename)
        # 从文件名提取币种名称
        coin_name = filename.replace('_features.json', '')
        # 特殊处理可能存在的币安币名称变体 (如果之前有替换空格)
        if coin_name == 'BinanceCoin':
             coin_name = 'BNB' # 或者你实际使用的名称
        # 你可能需要根据你的命名规则调整这里，确保能匹配 COIN_NAMES 列表

        print(f"\n处理文件: {filename} (币种: {coin_name})")

        min_time = None
        max_time = None
        valid_timestamps = 0
        total_news = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    news_list = json.load(f)
                    total_news = len(news_list)
                    print(f"  共找到 {total_news} 条新闻记录。")

                    timestamps = []
                    invalid_count = 0
                    for i, item in enumerate(news_list):
                        try:
                            # 尝试解析时间戳
                            ts_str = item.get('published_ts')
                            if ts_str:
                                # 尝试多种可能的格式，包括带时区和不带时区的
                                dt = pd.to_datetime(ts_str, errors='coerce')
                                if pd.notna(dt):
                                    timestamps.append(dt)
                                else:
                                    invalid_count += 1
                                    # print(f"    记录 {i+1}: 无效时间戳格式 '{ts_str}'")
                            else:
                                invalid_count += 1
                                # print(f"    记录 {i+1}: 缺少 'source_created_time' 字段")

                        except Exception as e:
                            invalid_count += 1
                            # print(f"    记录 {i+1}: 解析时间戳时出错 - {e}")
                            continue # 跳过这条记录

                    if invalid_count > 0:
                         print(f"  警告: {invalid_count} 条记录的时间戳无效或缺失。")

                    if timestamps:
                        valid_timestamps = len(timestamps)
                        min_time = min(timestamps)
                        max_time = max(timestamps)
                        results[coin_name] = {
                            'min_time': min_time,
                            'max_time': max_time,
                            'total_news': total_news,
                            'valid_timestamps': valid_timestamps
                        }
                        print(f"  有效时间戳数量: {valid_timestamps}")
                        print(f"  最早时间: {min_time}")
                        print(f"  最晚时间: {max_time}")
                        if min_time and max_time:
                             time_span_days = (max_time - min_time).days
                             print(f"  时间跨度: {time_span_days} 天")
                    else:
                         print("  错误: 未找到任何有效的 'published_ts'。")

                except json.JSONDecodeError:
                    print(f"  错误: 解析 JSON 文件失败: {file_path}")
                except Exception as e:
                    print(f"  处理文件时发生未知错误: {e}")

        except FileNotFoundError:
            print(f"  错误: 文件未找到: {file_path}")
        except Exception as e:
            print(f"  打开文件时发生错误: {e}")

    print("\n--- 汇总 ---")
    if results:
        for coin, data in results.items():
            span_days = (data['max_time'] - data['min_time']).days if data['min_time'] and data['max_time'] else 'N/A'
            print(f"币种: {coin:<5} | 最早: {data['min_time']} | 最晚: {data['max_time']} | 跨度(天): {span_days:<5} | 总新闻数: {data['total_news']:<6} | 有效时间戳: {data['valid_timestamps']:<6}")
    else:
        print("未能成功处理任何文件。")

if __name__ == "__main__":
    # *** 修改为你实际的 features 文件夹路径 ***
    features_directory = '/mnt/sda/ly/research/Project1/crypto_new_analyzer/features'
    check_news_time_span(features_directory)
