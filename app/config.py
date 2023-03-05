from pydantic import BaseSettings


class Settings(BaseSettings):
    http_schema: str = "http"
    time_format: str = "%H:%M"
    base_search_url: str = "https://www.google.com/complete/search"
    proxies_file: str = "proxies.txt"
    language: str = "en"
    client: str = "gws-wiz"
    today_format: str = "%Y-%m-%d"
    query_list: list = ['how', 'what', 'when', 'where', 'are', 'is', 'why', 'can']
    country_code: str = "LUX"
    request_headers: dict = {
        'authority': 'ogs.google.com',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Chrome OS"',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 14388.52.0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/98.0.4758.91 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,'
                  'application/signed-exchange;v=b3;q=0.9',
        'x-chrome-connected': 'source=Chrome,mode=0,enable_account_consistency=true,supervised=false,'
                              'consistency_enabled_by_default=false',
        'x-client-data': 'CJG2yQEIprbJAQipncoBCJbpygEIt6DLAQiTocsBCOvyywEInvnLAQiljswBCIGkzAE=',
        'sec-fetch-site': 'same-site',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'iframe',
        'referer': 'https://www.google.com/',
        'accept-language': 'en-US,en;q=0.9',
        # Requests sorts cookies= alphabetically
        # 'cookie': 'SID=KgipsdC3Yxa9m7JCFiSf9ERCriYX3tio-sdln7YpWH15A05EkNn3lzEaOxsxKB_j73HZ8Q.; __Secure-1PSID=KgipsdC3Yxa9m7JCFiSf9ERCriYX3tio-sdln7YpWH15A05EDjjIiujZe5PuNwI0jKTIUQ.; __Secure-3PSID=KgipsdC3Yxa9m7JCFiSf9ERCriYX3tio-sdln7YpWH15A05EgV9PEOK8KcjRZ-uQ-V6-3A.; HSID=A4EPy3YgwJXM51l6-; SSID=Ay5kAM2fDjIkfmpCX; APISID=Akv6x6eGTTq5dDC0/AMwo4EQGB5jsZRP11; SAPISID=Xdgah6CYZtLA1H0L/AlykPZRaBrmcFhebU; __Secure-1PAPISID=Xdgah6CYZtLA1H0L/AlykPZRaBrmcFhebU; __Secure-3PAPISID=Xdgah6CYZtLA1H0L/AlykPZRaBrmcFhebU; OTZ=6519179_34_34__34_; SEARCH_SAMESITE=CgQIwZUB; 1P_JAR=2022-05-25-06; NID=511=Vl8RZrmSnFRPwPAlCvStF4RZ5FRWrRCOngQ0jhVt8Lw5f-p8a-PyCbT_XdrJERuW932YfBn2AWGB6D_Msimu6gIBgM0eI47WAtWtHAiz6A3E3WoU8AaiRhsUlw3K9VBHTkS-DEXd6AEU96t60oe-ZIj9A9XVT1WKKIUdAsmHY5qhgRuqwst2Yif6s98aIFy8vMaoZ3K6rgU6hQCFbBU06M-hgy-ByRuvhph2V79vJBWXeT9Rcbh__pRmGnfv3K8Bh2oYngp2inVqlNxgxdqaFOvsrJWpFWIuC9JFJkCEAg; OGPC=19022519-1:; AEC=AakniGO1bomcWh2W_Hjby5dk15zIQE-zegTw55ljJQPb_rcF9eh_q2bycyM; SIDCC=AJi4QfHXY7354Db5dHXtuotnTXiDpiZeT500CGU_-gH7aV4aUGXrj6lieYU4l6U2tvWfhMWShA; __Secure-3PSIDCC=AJi4QfHU6C6zb0AcfqPm0l553V_uOB4y_wjFuhXD24na1ABfq53RoHU4_h1gW3rOAw0WAYzt6g',
    }
