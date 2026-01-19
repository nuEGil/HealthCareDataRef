import os
import sqlite3
import uvicorn
from pydantic import BaseModel 
from fastapi import FastAPI

'''fast api version of the ICD code look up
consider adding a queue up front soo requensts dont bounce.
this is going to be a local app, but I can simulate traffic 
later on. 

try out the service 
curl -X POST http://127.0.0.1:8001/search \
    -H "Content-Type: application/json" \
    -d '{"text": "opacity lung"}'

'''
app = FastAPI()

# Connect to the sql database 
con = sqlite3.connect(os.path.join(os.environ["SEARCH_DB_PATH"], "knowledge_store4.db")) # these are not thread safe
cur = con.cursor() # createa cursor object

class SearchRequest(BaseModel):
    text: str

# --- Search endpoint ---
@app.post("/search")
async def search(req: SearchRequest):
    # given a request of type Search Request, return the bottom result
    qwords = req.text.lower() # make it lower case  
    qwords = qwords.split(' ') # split by space. we can clean other characters later

    likes = " AND ".join(["lower(keywords) LIKE ?"] * len(qwords))

    sql = f"""
    SELECT wikititle, description, symptoms, keywords, google_term, icdcode
    FROM knowledge_store
    WHERE {likes}
    LIMIT 10
    """

    params = [f"%{q}%" for q in qwords]
    cur.execute(sql, params)
    rows = cur.fetchall()

    if not rows:
        return {"result": "<i>No results found</i>"} 

    html = "<b>Results</b><br/>"
    for wikititle, desc, symptoms, kwords, gterm, icdcode in rows:
        html += f"""
            <div class="wiki-entry">
                <h3 class="wiki-title">{wikititle}</h3>
                <p><b>Description:</b><br/>
                    {desc}
                </p>

                <p><b>Symptoms:</b><br/>
                    {symptoms}
                </p>
                
                <p><b>ICD Code Prefix:</b><br/>
                    {icdcode}
                    
                </p>

                <p><i>Enter {icdcode} on search page for full code </i><br/>
                    
                    
                </p>
                
                <p><b>Keyword tags:</b><br/>
                    {kwords}
                </p>

                <p><b> ICD term used to search Google:</b><br/>
                    {gterm}
                </p>


                <p><i>Disclaimer:</i><br/>
                database built with google search api + wikipedia rest api. visit the wikipedia page in the title for the full information. 
                </p>
            </div>
            """

    return {"result": html}

if __name__ == "__main__":
    uvicorn.run("app.services.knowledge_store_search:app", host="127.0.0.1", port=8001)
    con.close()  # close the sonnection to the database.