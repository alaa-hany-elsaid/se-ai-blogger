from datetime import datetime
import requests
import json
import html.parser
import socket
from pytrends.request import TrendReq

from app.config import Settings

settings = Settings()


class _HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super(_HTMLTextExtractor, self).__init__()
        self.result = []

    def handle_data(self, d):
        self.result.append(d)

    def get_text(self):
        return ''.join(self.result)


def _html_to_text(html_output):
    s = _HTMLTextExtractor()
    s.feed(html_output)
    return s.get_text()


def _text_to_questions(text):
    return [q for q in list(json.loads(text.encode('utf-8').decode('unicode-escape'))) if
            isinstance(q, list)]


counter = 1
now_time = datetime.now().strftime(settings.time_format)
req_time = int(now_time[:2]) * 60 + int(now_time[3:])


def search(search_query):
    global settings
    query_list = settings.query_list
    search_query_list = []
    for i in query_list:
        search_query_list.append(i + " " + search_query)
    for i in range(97, ord('z') + 1):
        search_query_list.append(search_query + ' ' + chr(i))

    headers = settings.request_headers
    out = {}
    for query_index in range(len(search_query_list)):
        params = {
            'q': search_query_list[query_index],
            'cp': 3,
            'client': settings.client,
            'hl': "{0}-{1}".format(settings.language, settings.country_code),
            'authuser': 0,
            'dpr': 1,
        }
        resp = requests.get(settings.base_search_url, params=params, headers=headers)
        try:
            final_tuple = _text_to_questions(_html_to_text(resp.text.replace("window.google.ac.h(", "")[:-1]))
        except:
            continue
        list_of_results = [result[0] for result in final_tuple[0]]
        for values in list_of_results:
            sp_val = values.split(' ')
            new = ' '.join(sp_val)
            if new[-4:] != '':
                list_of_results[list_of_results.index(values)] = new
            else:
                list_of_results[list_of_results.index(values)] = new[:-4]

        list_of_results = list(set(list_of_results))
        list_of_results.sort()
        qer = ' '.join(search_query_list[query_index].split(search_query)).strip()
        if len(list_of_results) > 0 and qer in list_of_results[0]:
            out[search_query_list[query_index]] = list_of_results

    return out


def get_trends(search_query):
    out = {}
    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=2, backoff_factor=0.1)

    except Exception as e:
        print(e)
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25),
                            proxies=[p.strip() for p in open(settings.proxies_file, 'r')].append(
                                settings.http_schema + '://' + socket.gethostbyname(socket.gethostname())), retries=3,
                            backoff_factor=0.1,
                            requests_args={'verify': False})

    kw_list = [search_query]  # list of keywords to get data
    pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
    try:
        data_for_related_topics = pytrends.related_topics()
        rising_title = list(data_for_related_topics[search_query]['rising']['topic_title'])
        rising_type = list(data_for_related_topics[search_query]['rising']['topic_type'])
        out["Related Topic Titles And Types With Rising Interest"] = [rising_title[i] + ' - ' + rising_type[i] for i in
                                                                      range(len(rising_title))]
        top_title = list(data_for_related_topics[search_query]['top']['topic_title'])
        top_type = list(data_for_related_topics[search_query]['top']['topic_type'])
        out["Top Searched Related Topic Titles and Types"] = [top_title[i] + ' - ' + top_type[i] for i in
                                                              range(len(top_title))]
    except:
        pass

    try:
        data_for_related_queries = pytrends.related_queries()
        rising_query = list(data_for_related_queries[search_query]['rising']['query'])
        rising_query_val = list(data_for_related_queries[search_query]['rising']['value'])
        top_query = list(data_for_related_queries[search_query]['top']['query'])
        top_query_val = list(data_for_related_queries[search_query]['top']['value'])
        out["Related Queries With Rising Interest And Their Search Values"] = [rising_query[i] + ' - ' + str(
            rising_query_val[i]) for i in range(len(rising_query))] + [top_query[i] + ' - ' + str(top_query_val[i]) for
                                                                       i in range(len(top_query))]
    except:
        pass
    data_over_time = pytrends.interest_over_time().reset_index()
    out["Keyword Web Search Interest Over Time"] = [
        {"date": str(data_over_time['date'][i].date()), "search_query": search_query,
         "isPartial": bool(data_over_time["isPartial"][i])} for i in range(len(data_over_time.index))]
    return out


