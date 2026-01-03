import os
import json 
import time
import requests
from bs4 import BeautifulSoup

'''
ok. so then the next part. you just need to get the article names..
seems like a job for the google custom search engine caller. 
so search engine request to find all the urls, then run the 
wiki caller. ... interesting. 
'''

if __name__ =='__main__':
    url = "https://en.wikipedia.org/"
    page = "Glucose"
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{page}"

    user_agent = os.environ['WIKI_USER_AGENT']
    headers = {
        "User-Agent": f"wikireader000/1.0 ({user_agent})",
        "Accept": "application/json"
    }
    html = requests.get(url, headers=headers).text
    # print(html)
    soup = BeautifulSoup(html, "html.parser")

    # remove junk
    for tag in soup(["sup", "table", "style"]):
        tag.decompose()

    text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    print(text)