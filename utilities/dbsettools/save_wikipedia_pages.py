import os
import json 
import time
import sqlite3
import requests
from bs4 import BeautifulSoup

"""
probably want to split this 
"""
class GoogleSearchTool():
    def __init__(self):
        self.url = "https://www.googleapis.com/customsearch/v1"
        self.params = {
            'key': os.environ['GOOGLE_SEARCH_API'],
            'cx':  os.environ['GOOGLE_SEARCH_ENGINE_ID'],
            'q':  ""}
        
        # probably do this as an argparse later. 
        self.data_path ="data/google_results/"
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def call(self, query, tag=0):
        self.params.pop('q')
        self.params['q'] = f"site:wikipedia.org {query}"    
        response = requests.get(self.url, params=self.params)

        if response.status_code == 200:
            data = response.json()
            print("Search successful!")
            print(data)
            # # Example: Print the title of the first result
            # if "items" in data:
            #     print(f"Top Result: {data['items'][0]['title']}")
            output_path = os.path.join(self.data_path,"google_search_id-{:05d}.json".format(tag))
            with open(output_path, "w") as f:
                json.dump(data, f, indent = 4)
        else:
            print(f"Error {response.status_code}: {response.text}")
        
        if 'items' in data.keys():
            output = data['items'][0]['link'].split('/')[-1]
            print("top google search term: ", output)
            return output # and thats the page name
        else: 
            return 0
class WikipediaRestAPITool():
    def __init__(self):        
        self.headers = {
            "User-Agent": f"{os.environ['WIKI_APP_NAME']}/1.0 \
                ({os.environ['WIKI_USER_AGENT']})",
            "Accept": "application/json"
        }
        self.url_base = f"https://en.wikipedia.org/api/rest_v1/page/html/" 
        
        # again make this with arg parse later. 
        self.data_path ="data/wikipedia_results/"
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
    
    def call(self, page, tag):
        url = self.url_base+page
        html = requests.get(url, headers=self.headers).text
        # save html to an html file
        output_name = os.path.join(self.data_path, "wikipedia_search_id-{:05d}.html".format(tag))
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(html)

def mainLoop():
    google = GoogleSearchTool()
    wiki = WikipediaRestAPITool()
    
    con = sqlite3.connect(os.environ["SEARCH_DB_PATH"]) # these are not thread safe
    cur = con.cursor() # createa cursor object    
    cur.execute("""
                SELECT short_desc
                FROM icd10cm
                WHERE LENGTH(CAST(code AS TEXT)) = 3
                LIMIT 10;
                """)
    rows = cur.fetchall()
    for i, row in enumerate(rows):
        print(row[0])
        top_page = google.call(query=row, tag=i)
        
        if not top_page==0:
            wiki.call(page = top_page, tag = i)        
        
        time.sleep(0.5)

        if i>10:
            break

    con.close()
    
    



if __name__ == '__main__':
    mainLoop()