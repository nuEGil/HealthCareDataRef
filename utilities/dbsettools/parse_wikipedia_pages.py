import os
import re
import glob
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
import math 
from bs4 import BeautifulSoup



'''
Need to refactor to clean this up. 

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

Using TFIDF for the keyword generation without the LLM -- scores dependant on document window contents. huh... context window again. 
May make a version of this with the gemini caller commands to add a second set of keywords
rememebr gemini gen ai only allows 20 calls per day. so be careful here. 


last step -- save entries to a json or directly to sql db... have to remember to make copeis where needed. 
'''

HARD_STOPWORDS = {
    "the","of","and","to","in","a","an","is","are","was","were",
    "for","on","with","as","by","at","from","that","this","these","those",
    "it","its","be","or","not","but","if","then","than","so",
    "can","may","might","should","would","could",
    "has","have","had","having",
    "such","other","same","both","each","per", "children"
}

def is_valid_token(t):
    return (
        t not in HARD_STOPWORDS
        and len(t) >= 3
        and not t.isdigit()
    )

@dataclass
class entry():
    google_term:str = field(default_factory=str)
    icdcode: str = field(default_factory=str)
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
            tokens = [
                        t for t in clean_text.split()
                        if is_valid_token(t)
                    ]

            term_counts = Counter(tokens)
            denom = sum(term_counts.values())
            for k,v in term_counts.items():
                self.term_frequencies[k] = v/denom
                self.tfidf[k] = 0.0

    def to_record(self):
        # make sure you run  the tfidf computer first. 
        return {
            "title": self.title, # wikipedia page title. 
            "description": self.description,
            "symptoms": self.symptoms,
            "keywords": self.keywords
        }

@dataclass
class TFIDF_computer():
    entries:list 
    doc_frequencies:dict  
    
    def compute_TFIDF(self): 
        N = len(self.entries)
        for aentry in self.entries:
            for term, tf_ in aentry.term_frequencies.items():
                # print('term :', term)
                df = self.doc_frequencies[term]
                idf = math.log((1 + N) / (1 + df)) + 1
                aentry.tfidf[term] = tf_ * idf 
            # after you get the  tfidf sort them
            top = sorted(aentry.tfidf.items(), key=lambda kv: (-kv[1], kv[0]))[0:10]
            # print(f"title:{aentry.title}, kw:{top}")
            aentry.keywords=",".join([t[0] for t in top])
            aentry.keywords+=f",{aentry.title}" # add the title into the key word search
                
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

def getDataEndID(path_, suffix = "html"):
    pages = glob.glob(os.path.join(path_, f"*.{suffix}"))
    return {p.split('-')[-1].split('.')[0]:p  for p in pages}

def readGoogleSearchJson(path_):
    with open(path_, "r", encoding ="utf-8") as ff:
        data= json.load(ff)
    gterm = data['queries']['request'][0]['searchTerms'][len("site:wikipedia.org")::]
    return gterm

def getIDsFromWiki(n_searches, start):
    dbpath = os.path.join(os.environ["SEARCH_DB_PATH"], "icd_10_codes.db")
    con = sqlite3.connect(dbpath) # these are not thread safe
    cur = con.cursor() # createa cursor object    
    # same query we used to run the google + wikipedia page search pipeline. 
    sql_ = """
            SELECT short_desc, code
            FROM icd10cm
            WHERE LENGTH(CAST(code AS TEXT)) = 3
            ORDER BY code
            LIMIT ? OFFSET ?;
            """
    cur.execute(sql_, (n_searches, start))
    rows = cur.fetchall()
    # same string formating as when we saved the wikipedia pages
    rows_ = {"{:05d}".format(ii):(r[0],r[1]) for ii, r in enumerate(rows)}
    return rows_

def GrabAllEntriesAndTFIDF():
    # Get known rows -- add arg parse for this. 
    pipeline_queries = getIDsFromWiki(n_searches=30, start=0)
    
    # # Get google and wiki paths.  
    # google_paths = os.path.join(os.environ["SEARCH_DB_PATH"], "base_crawl/google_results/")
    # google_pages = getDataEndID(google_paths, 'json')
    # print(google_pages.keys())

    wiki_paths = os.path.join(os.environ["SEARCH_DB_PATH"], "base_crawl/wikipedia_results/")
    wiki_pages = getDataEndID(wiki_paths)
    print(wiki_pages.keys())

    # wiki_paths has less pages -> not all the search results yield a wikipedia page. 
    all_entries = []
    doc_term_counter = Counter()
    # need one loop to get the entries and get the 
    for ii, (wiki_key, wiki_path) in enumerate(wiki_pages.items()):
        # print('wiki path     ',wiki_path)
        # google_term = readGoogleSearchJson(google_pages[wiki_key])
        pquery = pipeline_queries[wiki_key]
        # can do an assert google_term == pquery if you want. 

        # print('check ', google_term, pquery)


        # read the wikipedia page
        with open(wiki_path, "r", encoding ="utf-8") as f0:
            wiki_page = f0.read()
        print("page number: ", ii)
    
    
        new_entry = getEntry(wiki_page)
        new_entry.google_term = ""+pquery[0]
        new_entry.icdcode = ""+pquery[1]
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
    for j in range(len(coco.entries)):
        print(coco.entries[j].title, ": ",coco.entries[j].keywords)

    return coco

def buildKnowledgeStore(data_):
    # using basically the same commands from the parseICD10 script
    outname = os.path.join(os.environ["SEARCH_DB_PATH"], "knowledge_store4.db")
    
    con = sqlite3.connect(outname) # can store the icd10 CM + PCS together later idk
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_store (
    id INTEGER PRIMARY KEY,
    google_term TEXT,
    icdcode TEXT,
    wikititle TEXT,
    description TEXT,
    symptoms TEXT,
    keywords TEXT);
    """)

    for ie, ent in enumerate(data_.entries):
        command = """INSERT INTO knowledge_store (id,google_term,icdcode,wikititle,description,symptoms,keywords) 
        VALUES (?,?,?,?,?,?,?)"""
        # 1 indexed table
        cur.execute(command, (ie,
                              ent.google_term,
                              ent.icdcode,
                              ent.title,
                              ent.description,
                              ent.symptoms,
                              ent.keywords,))
        
    con.commit()# commit after insert. 

    # print some stuff so i know it worked
    cur.execute("SELECT * FROM knowledge_store LIMIT 10")
    for row in cur.fetchall():
        print(row)

    con.close() # close .db file
    return 0 # exit code 

if __name__ == '__main__':
   
    TFIDF_Comp_Data = GrabAllEntriesAndTFIDF()
    # next step is to put it all into an sql database. 
    buildKnowledgeStore(TFIDF_Comp_Data) # only run once. 