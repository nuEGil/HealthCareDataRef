import os
import re
import time
import glob
import sqlite3
from bs4 import BeautifulSoup
'''
get symptoms
get description
either add to icd10 db or make a new db 
copy the old db so you dont have to run the old code if you mess this up. 
'''

if __name__ == '__main__':
    
    pages = glob.glob(os.path.join("data/lungdat/wikipedia_results/", "*.html"))
    
    for page in pages:
        with open(page, "r") as f:
            file_ = f.read()
        soup = BeautifulSoup(file_, 'html.parser')
        title = soup.find_all(["title"])[0].text.strip()

        print('--------')
        print("title :",title)
        print('--------')
        # this thing will find headings. so you just need 
        heading_tags = ["p", "li", "h1", "h2", "h3"]
        cond = False
        symptoms_text =""

        # edge detect - store all text between symptoms headings    
        for tags in soup.find_all(heading_tags):
            if "h" in tags.name and "symptoms" in tags.text.strip().lower():
                cond=True

            if "h" in tags.name and not "symptoms" in tags.text.strip().lower():
                cond=False

            if cond:
                symptoms_text+=tags.text.strip()
                symptoms_text+="\n"
        
        print(symptoms_text)


        