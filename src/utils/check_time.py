import requests
import time
from datetime import datetime

time_format = "%y-%m-%d_%Hh-%Mm-%Ss"
time_format2 = "%Y %B %d, %I:%M %p"


def get_time(format: str = time_format):
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

# if __name__ == "__main__":
#     print(get_time(format=time_format2))
#
#     start_time = "23-10-04_14h-30m-45s"
#     end_time = "23-10-04_14h-31m-44s"
#     print(cal_time_diff(start_time, end_time))
