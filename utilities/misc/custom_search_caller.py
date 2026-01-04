import os
import json 
import time
import requests
'''turn this into an extract transform load process
search api gets 100 queries per day. so what I can do is 

1.  load the icd10 codes order.txt ,and grab 50 entries
2. query to get laymans terms for these things
3. query to get symptoms

Grab links --> 

the LLM assisted search tool for max project burn rate right 
1. use a gemini api call with a prompt to search for something
2. format the saerch to look on a list of known websites for this use case
3. call google search engine api to put links into the response text. 

list of sites
display links for the copyrighted sites -> point to the website only. 
mayoclinic.org  - copyrighted
medlineplus.gov - copyrighted  

- creative commons - more useful for other data. 
wikipedia.org  -- has a rest api. 

'''

def CustomSearchAPICall():
    url = "https://www.googleapis.com/customsearch/v1"
    # query = "site:medlineplus.gov medical term for ear infection"
    query = "site:wikipedia.org medical term for ear infection"
    # query = f"{title} common name symptoms site:medlineplus.gov OR site:wikipedia.org"
    params = {
        'key': os.environ['GOOGLE_SEARCH_API'],
        'cx':  os.environ['GOOGLE_SEARCH_ENGINE_ID'],
        'q':   query
    }
    # requests.get will do this part for you
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print("Search successful!")
        print(data)
        # # Example: Print the title of the first result
        # if "items" in data:
        #     print(f"Top Result: {data['items'][0]['title']}")
        with open("data/wiki-response_data.json", "w") as f:
            json.dump(data, f, indent=4)
    else:
        print(f"Error {response.status_code}: {response.text}")

# add this loop in 
def generate_local_knowledge_base(icd_list):
    for code, title in icd_list:
        file_path = f"data/raw_json/{code}.json"
        if os.path.exists(file_path):
            continue  # Skip if already downloaded
            
        # Refined query to get symptoms and names in one go
        query = f"{title} common name symptoms site:medlineplus.gov OR site:wikipedia.org"
        
        # Call your existing CustomSearchAPICall function here...
        # Save the raw response
        
        time.sleep(1) # Be nice to the API

def response_explore():
    with open("data/wiki-response_data.json", "r",  encoding = "utf-8") as ff:
        data = json.load(ff)
    
    print(type(data)) 

if __name__ == '__main__':
    # CustomSearchAPICall()
    response_explore()