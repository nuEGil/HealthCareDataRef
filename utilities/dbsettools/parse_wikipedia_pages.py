import os
import re
import glob
import json
import copy
from collections import Counter
from dataclasses import dataclass, field, asdict
import math 

from bs4 import BeautifulSoup
'''
get symptoms
get description

Paths for this code.
1. is if we have the search terms we used that were reading DB entries, then we can just read the wikipedia pages directly and add to a new database with symptoms
2. there's a version for the demo where we read a set list of lung pathologies. we need different logic for that. 

LogicA - icd10 to google search to wiki pages
1. read google search jsons for the search terms. 
2. parse the wikipedia pages 
3. output results to sql database so it has - icd term + symptoms, + wiki page

LogicB - set list of terms to google search to wiki pages
1. get keywords especially symptoms from any wikipedia pages
2. search icd10 database for a match given the key words


but you still need to keep the same format... gonna need a data class. 


ranking # term_frequency = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))  # count desc, token asc)
for tfidf batch it  - might have a wikipedia results dir that's too big. 
that's interesting. the batching part 
'''
@dataclass
class entry():
    title: str = field(default_factory=str)
    description: str = field(default_factory=str)
    symptoms: str = field(default_factory=str)
    keywords: str = field(default_factory=str)

    # --- some parts to do TF IDF -- do not store these counters in db
    term_frequencies: dict = field(default_factory=Counter) 
    tfidf: dict = field(default_factory=dict)     
    
    def computeTermFrequency(self):
        if len(self.term_frequencies) == 0:
            pooled_text = self.symptoms + " " + self.description
            clean_text = re.sub(r'[^0-9a-z]+', ' ', pooled_text.lower())
            term_counts = Counter(clean_text.split())
            denom = sum(term_counts.values())
            for k,v in term_counts.items():
                self.term_frequencies[k] = v/denom
                self.tfidf[k] = 0.0

@dataclass
class TFIDF_computer():
    entries:list 
    doc_frequencies:dict  
    
    def compute_TFIDF(self): 
        N = len(self.entries)
        for aentry in self.entries:
            for term, tf in aentry.term_frequencies.items():
                # print('term :', term)
                df = self.doc_frequencies[term]
                idf = math.log((1 + N) / (1 + df)) + 1
                aentry.tfidf[term] = math.log10(tf) + idf
            # after you get the  tfidf sort them
            top = sorted(aentry.tfidf.items(), key=lambda kv: (-kv[1], kv[0]))[0:10]
            print(f"title:{aentry.title}, kw:{top}")
            aentry.keywords=",".join([t[0] for t in top])
            aentry.keywords+=f",{aentry.title}"
            # aentry.sortTFIDF()
                
def getEntry(file_):
    page_entrty = entry() # make an entry
    
    soup = BeautifulSoup(file_, 'html.parser')
    title = soup.find_all(["title"])[0].text.strip()
    page_entrty.title = title
    
    # this thing will find headings. so you just need 
    heading_tags = ["title", "p", "li", "h1", "h2", "h3"]
    desc_cond = False
    description_text =""
    
    symptom_cond = False
    symptoms_text = ""
    
    # edge detect - store all text between symptoms headings    
    for tags in soup.find_all(heading_tags):
        # -- Check for description first ---
        if tags.name == "title":
            desc_cond=True

        if "h" in tags.name:
            desc_cond=False

        # -- check for symptoms --
        if "h" in tags.name and "symptoms" in tags.text.strip().lower():
            symptom_cond=True

        if "h" in tags.name and not "symptoms" in tags.text.strip().lower():
            symptom_cond=False
            
    
        # Grab the description
        if desc_cond and not symptom_cond:
            description_text+=tags.text.strip()
            description_text+="\n"

        # grab the symptoms 
        if symptom_cond and not desc_cond:
            symptoms_text+=tags.text.strip()
            symptoms_text+="\n"

        page_entrty.description = description_text
        page_entrty.symptoms = symptoms_text

    return page_entrty


if __name__ == '__main__':
    lung_dat_path = os.path.join(os.environ["SEARCH_DB_PATH"], "lungdat/wikipedia_results/")
    # lung_dat_path = os.path.join(os.environ["SEARCH_DB_PATH"], "base_crawl/wikipedia_results/")
    
    pages = glob.glob(os.path.join(lung_dat_path, "*.html"))
    
    all_entries = []
    doc_term_counter = Counter()
    # need one loop to get the entries and get the 
    for ii, page in enumerate(pages):
        print("page number: ", ii)
        with open(page, "r") as f:
            file_ = f.read()
    
        new_entry = getEntry(file_)
        print(new_entry.title)
        print(f"len desc :{len(new_entry.description)}  len symptoms: {len(new_entry.symptoms)}")
        print(new_entry)
        print('\n')

        new_entry.computeTermFrequency() 
        for term in new_entry.term_frequencies:
            doc_term_counter[term]+=1
        all_entries.append(new_entry)

    # print(doc_term_counter)
    
    coco = TFIDF_computer(entries=all_entries, doc_frequencies=doc_term_counter)
    coco.compute_TFIDF()
    print(coco.entries[0].title, ": ",coco.entries[0].keywords)
