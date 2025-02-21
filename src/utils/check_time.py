import requests
import time
from datetime import datetime

def get_time():
    try:
        url = 'https://sapi.k780.com'
        request_result = requests.get(url=url)
        if request_result.status_code == 200:
            headers = request_result.headers
            net_date = headers.get("date")
            gmt_time = time.strptime(net_date[5:25], "%d %b %Y %H:%M:%S")
            bj_timestamp = int(time.mktime(gmt_time) + 8 * 60 * 60)
            bj_timestamp =  datetime.fromtimestamp(bj_timestamp)
            return bj_timestamp.strftime("%y-%m-%d_%Hh-%Mm-%Ss")
    except Exception as exc:
        return datetime.now().strftime("%y-%m-%d_%Hh-%Mm-%Ss")

# if __name__ == "__main__":
#     print(get_time())