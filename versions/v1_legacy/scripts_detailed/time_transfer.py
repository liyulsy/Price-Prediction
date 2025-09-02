import datetime

def unix_timestamp_to_beijing_time(timestamp_ms):
    timestamp_sec = timestamp_ms / 1000
    utc_datetime = datetime.datetime.fromtimestamp(timestamp_sec, datetime.timezone.utc)
    beijing_timezone = datetime.timezone(datetime.timedelta(hours=8))
    beijing_datetime = utc_datetime.astimezone(beijing_timezone)

    return beijing_datetime.strftime('%Y-%m-%d %H:%M:%S')

# 示例用法
if __name__ == "__main__":
    timestamp_ms = 1710926422790  # Unix时间戳（毫秒）
    beijing_time = unix_timestamp_to_beijing_time(timestamp_ms)
    print(beijing_time)