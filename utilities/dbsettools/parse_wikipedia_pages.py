import os
import re
import time
import glob
import sqlite3
from bs4 import BeautifulSoup

if __name__ == '__main__':
    
    pages = glob.glob(os.path.join("data/lungdat/wikipedia_results/", "*.html"))
    with open(pages[0], "r") as f:
        file_ = f.read()
    
    soup = BeautifulSoup(file_, 'html.parser')
   
    # this thing will find headings. so you just need 
    heading_tags = ["p", "li", "h1", "h2", "h3"]
    for tags in soup.find_all(heading_tags):
        print(tags.name + ' -> ' + tags.text.strip())
    