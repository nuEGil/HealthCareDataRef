import os 
import sqlite3
from services.base_interface import SearchService

class LocalSearchService(SearchService):
    def __init__(self, db_conn):        
        self.cur = db_conn.cursor()
        self.db_conn = db_conn

    def search(self, query: str) -> str:
        # print('querry : ', query)
        sql = """
        SELECT code, short_desc, long_desc
        FROM icd10cm
        WHERE long_desc LIKE ?
        LIMIT 10
        """

        self.cur.execute(sql, (f"%{query}%",))
        rows = self.cur.fetchall()

        if not rows:
            return "<i>No results found</i>"

        html = "<b>Results</b><br/>"
        for code, short_desc, long_desc in rows:
            html += (
                f"<p>"
                f"<b>{code}</b>: {short_desc}<br/>"
                f"{long_desc}"
                f"</p>"
            )

        return html