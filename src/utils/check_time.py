
import json
import requests
import time

from datetime import datetime, timezone, timedelta
from loguru import logger

time_format = "%y-%m-%d_%Hh-%Mm-%Ss"  # 25-03-04_15h-00m-01s
time_format1 = "%Y %B %d, %I:%M %p"  # 2025 March 04, 03:02 PM
time_format2 = "%y%b %d_%H-%M-%S"  # 25Mar 04_15-00-02  " 0" will be replaced
time_format3 = "%I:%M %p"  # 03:01 PM


def get_time(format: str = time_format):
    apis = [
        tencen_time,
        suning_time,
        k780_time
    ]
    for api in apis:
        time = api(format)
        if time != None:
            return time
    return datetime.now().strftime(format)


def cal_time_diff(start_time, end_time, format=time_format):
    """
    calculate_time_difference
    :param start_time:
    :param end_time:
    :return: time_difference
    """
    start_time = datetime.strptime(start_time, format)
    end_time = datetime.strptime(end_time, format)

    if end_time < start_time:
        return "Error: end time is earlier than start time."

    time_difference = end_time - start_time

    days = time_difference.days
    seconds = time_difference.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days} days {hours} hours {minutes} minutes {seconds} seconds"
    elif hours > 0:
        return f"{hours} hours {minutes} minutes {seconds} seconds"
    elif minutes > 0:
        return f"{minutes} minutes {seconds} seconds"
    else:
        return f"{seconds} seconds"


def get_main_time():
    return get_time(format=time_format2).replace(" 0", "").replace(" ", "")


def k780_time(format: str):
    try:
        url = 'https://sapi.k780.com'
        request_result = requests.get(url=url)
        if request_result.status_code == 200:
            headers = request_result.headers
            net_date = headers.get("date")
            gmt_time = time.strptime(net_date[5:25], "%d %b %Y %H:%M:%S")
            bj_timestamp = int(time.mktime(gmt_time) + 8 * 60 * 60)
            bj_timestamp = datetime.fromtimestamp(bj_timestamp)
            return bj_timestamp.strftime(format)
    except Exception as exc:
        logger.error(f"K780 time error: {exc}")
        return None


def tencen_time(format: str):
    try:
        response = requests.get('http://vv.video.qq.com/checktime?otype=json')
        if response.status_code == 200:
            response_text = response.text
            if response_text.startswith("QZOutputJson="):
                json_str = response_text[len("QZOutputJson="):].rstrip(';')
                data = json.loads(json_str)
                timestamp = data['t']
                time_utc = datetime.utcfromtimestamp(timestamp)
                local_timezone = timezone(timedelta(hours=8))
                time = time_utc.replace(tzinfo=timezone.utc).astimezone(local_timezone)
                return time.strftime(format)
    except Exception as exc:
        logger.error(f"Tencent time error: {exc}")
        return None


def suning_time(format: str):
    try:
        response = requests.get('https://f.m.suning.com/api/ct.do')
        if response.status_code == 200:
            data = response.json()
            timestamp = data['currentTime'] / 1000
            time_utc = datetime.utcfromtimestamp(timestamp)
            local_timezone = timezone(timedelta(hours=8))
            time = time_utc.replace(tzinfo=timezone.utc).astimezone(local_timezone)
            return time.strftime(format)
    except Exception as exc:
        logger.error(f"Suning time error: {exc}")
        return None


# if __name__ == "__main__":
#     # print(get_time(format=time_format))
#     # print(get_time(format=time_format1))
#     # print(get_time(format=time_format2))
#     # print(get_time(format=time_format3))
#     #
#     # test = "25Mar 10_14-46-21"
#     # print(test.replace(" 0", "").replace(" ", ""))
#     #
#     # start_time = "23-10-04_14h-30m-45s"
#     # end_time = "23-10-04_14h-31m-44s"
#     # print(cal_time_diff(start_time, end_time))
#     time = k780_time(time_format3)
#     print(time)
#
#     time = tencen_time(time_format3)
#     print(time)
#
#     time = suning_time(time_format3)
#     print(time)
