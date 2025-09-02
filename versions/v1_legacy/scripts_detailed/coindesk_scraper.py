import requests
import json
from datetime import datetime
import os
import time

def fetch_crypto_news(keyword, to_ts, max_retries=3):
    """
    爬取Coindesk加密货币新闻
    :param keyword: 加密货币名称
    :param to_ts: 截止时间戳
    :param max_retries: 最大重试次数
    :return: 新闻数据列表
    """
    url = "https://data-api.coindesk.com/news/v1/search"
    params = {
        "search_string": keyword,
        "lang": "EN",
        "source_key": "coindesk",
        "limit": 100,
        "to_ts": to_ts
    }
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get("Data", [])
                if not articles:
                    print("没有找到相关文章。")
                    return []
                print(f"=== 关于\"{keyword}\"的最新新闻，共 {len(articles)} 篇 ===")
                return articles
            else:
                print(f"请求失败: {response.status_code}，正在重试...")
        except requests.exceptions.RequestException as e:
            retry_count += 1
            print(f"请求异常: {e}，正在第{retry_count}次重试...")
            time.sleep(2)
    print(f"多次重试（{max_retries}次）后仍然失败，请检查网络环境。")
    return []

def save_articles(keyword, new_articles):
    save_dir = "crypto_news"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_filename = f"{save_dir}/{keyword}.json"

    # 读取原有数据
    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as f:
            all_articles = json.load(f)
    else:
        all_articles = []

    # 合并并去重（以ID为唯一标识）
    existing_ids = {article["ID"] for article in all_articles}
    for article in new_articles:
        if article["ID"] not in existing_ids:
            all_articles.append(article)
            existing_ids.add(article["ID"])

    # 保存合并后的数据
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    print(f"已累计保存 {len(all_articles)} 条 {keyword} 新闻到 {json_filename}")
    return len(all_articles)  # 返回当前总数

if __name__ == "__main__":
    keyword = 'XRP, Ripple'
    to_ts = 1746604003
    stop_ts = 1640907140
    max_news_count = 6000
    while True:
        articles = fetch_crypto_news(keyword, to_ts, max_retries=10)
        if not articles:
            print("没有更多新闻，程序结束。")
            break
        # 保存到同一个文件
        total_count = save_articles(keyword, articles)
        if total_count >= max_news_count:
            print(f"新闻数量已达到设定上限（{max_news_count}条），程序结束。")
            break
        # 取本次最早一条新闻的PUBLISHED_ON
        min_ts = min(int(article['PUBLISHED_ON']) for article in articles)
        # if min_ts < stop_ts:
        #     print(f"已到达设定停止时间，程序结束。")
        #     break
        to_ts = min_ts - 1
        print(f"下次爬取将使用to_ts={to_ts}")
        time.sleep(1)  # 可根据需要调整间隔