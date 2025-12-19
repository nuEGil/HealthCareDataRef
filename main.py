import os 
import csv 

MIMIC_DIR = os.environ["MIMIC_DIR"]

def ImportMimicData():
    files = [
        "diagnosis.csv",
        "edstays.csv",
        "medrecon.csv",
        "pyxis.csv",
        "triage.csv",
        "vitalsign.csv",
        ]
    
    # so then you write a for loop to ingest each file. 
    fname = os.path.join(MIMIC_DIR, files[0])
    ci = 0
    with open(fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:
            print(', '.join(row), len(row))
            ci+=1
            if ci>10:
                break        
                
if __name__ =='__main__':
    ImportMimicData()
    