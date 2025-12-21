import os 
import sqlite3
from services.base_interface import SearchService

class LocalSearchService(SearchService):
    def __init__(self, db_conn):        
        self.cur = db_conn.cursor()
        self.db_conn = db_conn

    def search(self, query: str) -> str:
        self.cur.execute("SELECT * FROM icd10cm LIMIT 10")
        for row in self.cur.fetchall():
            print(row)

        return f"<b>Local result</b><br/>Query was: {query}"