import os  
import sqlite3
from dataclasses import dataclass

"""Read the OrderFiles.pdf there is some formating shared beteween 
Code Modification (CM) and Procedure coding system (PCS) files 
so a data class might be useful here - but only to store the format 
and to store the entries. 

Things to add 
1. xml parser 
2. some tools to update the different parts of the database. Code Descriptions folder has everything, 
but the table and index (xml files) has quick jumps good for human reference. 

"""

@dataclass
class ICDTable:
    code_type: str
    entries: dict

# in memory dictionary
def ParseICDOrder(fname):
    data = ICDTable(code_type='ICD_10_CM', 
             entries= {
                        'ord_number': [],
                        'code': [],
                        # Header = 0,1 int if valid HIPAA transaction
                        'header': [], 
                        'short_desc': [],
                        'long_desc': [],
                    }
             )
    # think about weaving the sql database update calls here. this data is small enough 
    # that you dont have to execute everything as you read a line... 
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # order table is one indexed
            data.entries['ord_number'].append(line[0:5].strip()) # pos 1 len 5
            data.entries['code'].append(line[6:13].strip()) # pos 7 len 7
            data.entries['header'].append(line[14:15].strip()) # pos 15 len 1
            data.entries['short_desc'].append(line[16:76].strip()) # pos 17 len 60
            data.entries['long_desc'].append(line[77:].strip()) # pos 77 to end long desc
            break

    print(data)              
    return data

# build an sqlite database without needed to keep the whole text file in memory... 
def buildICDDB(fname, outname):
    con = sqlite3.connect(outname) # can store the icd10 CM + PCS together later idk
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS icd10cm (
    id INTEGER PRIMARY KEY,
    code TEXT,
    header INT,
    short_desc TEXT,
    long_desc TEXT);
    """)

    # think about weaving the sql database update calls here. this data is small enough 
    # that you dont have to execute everything as you read a line... 
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            command = """INSERT INTO icd10cm (id,code,header,short_desc,long_desc) 
            VALUES (?,?,?,?,?)"""
            # 1 indexed table
            cur.execute(command, (int(line[0:5].strip()),# ordernum pos 1 len 5
                                  line[6:13].strip(),# icd cm/pcs code pos 7 len 7
                                  int(line[14:15].strip()),# header pos 15 len 1
                                  line[16:76].strip(), # short desc pos 17 len 60
                                  line[77:].strip()  # long desc pos 77 to end long desc
                                  ))
    con.commit()# commit after insert. 

    # print some stuff so i know it worked
    cur.execute("SELECT * FROM icd10cm LIMIT 10")
    for row in cur.fetchall():
        print(row)

    con.close() # close .db file
    return 0 # exit code for myself

if __name__ == '__main__':
    ICD10DIR = os.environ["CDC_ICD_10_DIR"]

    ord_file = os.path.join(ICD10DIR, "icd10cm-Code Descriptions-2026/icd10cm-order-2026.txt")   
    buildICDDB(ord_file, "data/icd_10_codes.db") #only run on build - do something else for updating the database. 
    # ParseICDOrder(ord_file)