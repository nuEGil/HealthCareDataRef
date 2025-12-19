import os 
import csv 
import sqlite3
from dataclasses import dataclass
'''radd in row numbers later. '''

MIMIC_DIR = os.environ["MIMIC_DIR"]

@dataclass
class TableHeads:
    title: str
    filename: str
    heads: dict

def ParseCSV(fname):
    # hold the whole csv data in memory i guess. 
    data = []
    with open(fname, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for ir, row in enumerate(datareader):
            data.append(row)
            # print('row len ', len(row))
            if ir == 0:
                header = tuple(row)
                # print("header: ", header)
    # print(f"fname: {fname} len: {len(data)}") 
    print('---')              
    return data

def ImportMimicData():
    # then the other part is to only run this if the data base does not exist right. 
   
    tables = [
                TableHeads(title = "diagnosis",
                    filename=os.path.join(MIMIC_DIR,"diagnosis.csv"),
                    heads = {'subject_id':"INT", 
                            'stay_id':"INT", 
                            'seq_num':"INT", 
                            'icd_code':"TEXT", 
                            'icd_version':"TEXT", 
                            'icd_title':"TEXT"}),
    
                TableHeads(title = "medrecon",
                        filename=os.path.join(MIMIC_DIR,"medrecon.csv"),
                        heads = {'subject_id':"INT", 
                                    'stay_id':"INT", 
                                    'charttime':"TEXT", 
                                    'name':"TEXT", 
                                    'gsn':"INT", 
                                    'ndc':"TEXT", 
                                    'etc_rn':"INT", 
                                    'etccode':"TEXT", 
                                    'etcdescription':"TEXT"})

                ]
    con = sqlite3.connect("mimic_data.db")
    cur = con.cursor()
    
    for th in tables:
        data = ParseCSV(th.filename)
        print(data[1])
        print('heads length', len(th.heads))
        ## sqlite 3 commands and stuff
        # start a sqlite data base
        
        sql_command = f"CREATE TABLE IF NOT EXISTS {th.title} ("
        for k,v in th.heads.items():
            sql_command+=f"{k} {v},"
        sql_command = sql_command[0:-1]
        sql_command+=");"
        print(sql_command)

        cur.execute(sql_command)

        columns = ",".join(th.heads.keys())
        placeholders = ",".join(["?"] * len(th.heads))

        cur.executemany(
            f"INSERT INTO {th.title} ({columns}) VALUES ({placeholders})",
            [row[:len(th.heads)] for row in data[1:]]
            )
        con.commit()# commit after insert. 
    con.close() # close .db file

    ## updating these next. 
    # TableHeads(title = "edstays",
    #            heads = {'subject_id':"INT", 
    #                     'hadm_id':"INT", 
    #                     'stay_id':"INT", 
    #                     'intime':"TEXT", 
    #                     'outtime':"TEXT", 
    #                     'gender':"TEXT", 
    #                     'race':"TEXT", 
    #                     'arrival_transport':"TEXT", 
    #                     'disposition':"TEXT"})
    
    # TableHeads(title="pyxis",
    #            heads={'subject_id':"INT", 
    #                   'stay_id':"INT", 
    #                   'charttime':"TEXT", 
    #                   'med_rn':"INT", 
    #                   'name':"TEXT", 
    #                   'gsn_rn':"INT", 
    #                   'gsn':"INT"})
    
    # TableHeads(title="triage",
    #            heads={'subject_id':"INT", 
    #                   'stay_id':"INT", 
    #                   'temperature':"FLOAT", 
    #                   'heartrate':"INT", 
    #                   'resprate':"INT", 
    #                   'o2sat':"INT", 
    #                   'sbp':"INT", 
    #                   'dbp':"INT", 
    #                   'pain':"TEXT", 
    #                   'acuity':"INT", 
    #                   'chiefcomplaint':"TEXT"})
    
    # TableHeads(title="vitalsign",
    #            heads={'subject_id':"INT", 
    #                   'stay_id':"INT", 
    #                   'charttime':"TEXT", 
    #                   'temperature':"FLOAT", 
    #                   'heartrate':"INT", 
    #                   'resprate':"INT", 
    #                   'o2sat':"INT", 
    #                   'sbp':"INT", 
    #                   'dbp':"INT", 
    #                   'rhythm':"TEXT", 
    #                   'pain':"TEXT"})
   
if __name__ =='__main__':
    ImportMimicData()
    