import os
import sqlite3
import uvicorn
from pydantic import BaseModel 
from fastapi import FastAPI
from fastapi.responses import FileResponse

'''fast api version of the ICD code look up
consider adding a queue up front soo requensts dont bounce.
this is going to be a local app, but I can simulate traffic 
later on. 

try out the service 
curl -X POST http://127.0.0.1:8000/search \
    -H "Content-Type: application/json" \
    -d '{"text": "diabetes"}'
'''
app = FastAPI()

# Connect to the sql database 
con = sqlite3.connect(os.path.join(os.environ["SEARCH_DB_PATH"], "icd_10_codes.db")) # these are not thread safe
cur = con.cursor() # createa cursor object

class SearchRequest(BaseModel):
    text: str

# --- Search endpoint ---
@app.post("/search")
async def search(req: SearchRequest):
    # given a request of type Search Request, return the bottom result
    keyword = req.text.lower() # make it lower case  
    keywords = keyword.split(' ') # split by space. we can clean other characters later

    if keywords[0] == "code":
        # only use the 1st ICD code following the keyword to search
        sql = f"""
            SELECT code, short_desc, long_desc
            FROM icd10cm
            WHERE code LIKE ?
            """

        cur.execute(sql, (f"{keywords[1].upper()}%",))
        rows = cur.fetchall()

    elif keywords[0] == "terms":
        # also the base case
        likes = " OR ".join(["LOWER(long_desc) LIKE ?"] * len(keywords[1::]))
        sql = f"""
        SELECT code, short_desc, long_desc
        FROM icd10cm
        WHERE {likes}
        """

        params = [f"%{t}%" for t in keywords[1::]]
        cur.execute(sql, params)
        rows = cur.fetchall()

        results = []
        # this matches similarity. need to add a loop to drop results. 
        for r in rows:
            score = len(set(keywords[1::]) & set(r[2].lower().split()))
            if score > 0:
                results.append((score, r))
            
        results.sort(reverse=True, key=lambda x: x[0])
        rows = [r[1] for r in results]
        rows = rows[0:20]  # need to add in a slider or something to change num results 

    else:
        rows =[]


    if len(rows)<1:
        return {"result": "<i>No results found</i>"} 

    html = "<b>Top (20) Results</b><br/>"
    for code, short_desc, long_desc in rows:
        html += (
            f"<p>"
            f"<b>{code}</b>: {short_desc}<br/>"
            f"{long_desc}"
            f"</p>"
        )

    return {"result": html}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    con.close()  # close the sonnection to the database.