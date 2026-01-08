import os
import csv
import sys 
import glob
import sqlite3
from array import array
from dataclasses import dataclass, field

'''
make an sql database from the provided datasheets
sample the sql database into a csv when you want to 
make training/ testing data sets. 

more work up front by implpementing data classes but 
when we want to pull a sub set, it'll pay off. 

add an argparse 
1. start database
2. get sample classes of interest - split list of words, and run sql queries to return subsets 
'''
# data classes 
@dataclass
class entry_batch:
    image_ind: str = field(default_factory=str)
    path: str = field(default_factory=str)
    batch: str = field(default_factory=str)
    
@dataclass
class entry_BBoxList2017:
    image_ind: str=field(default_factory=str)
    finding_label: str=field(default_factory=str)
    bboxstr: array=field(default_factory = lambda:array('d',[0.0, 0.0, 0.0, 0.0]))

@dataclass 
class entry_dataEntry2017:
    image_ind:     str=field(default_factory=str)
    finding_label: str=field(default_factory=str) # string sep | for multiple findings
    follow_up_num: int=field(default_factory=int)	
    patient_id:    int=field(default_factory=int) # ~30k patients 	
    patient_age:   int=field(default_factory=int)
    patient_gender:	str=field(default_factory=str)
    view_position:  str=field(default_factory=str) # Post anterior(PA ->back to front) Anteroposterior(AP,front back)
    # images are resized to 1024x1024 from original width, height --> spacing is the image resolution 
    # keeping both as floats... math later. 
    originalImage_wh: array=field(default_factory = lambda:array('d',[0.0, 0.0])) # int in spread sheet
    originalImage_xyres: array=field(default_factory = lambda:array('d',[0.0, 0.0])) # float in spread sheet

# Readers
def GetImgPaths():
    data_dir = os.environ['CHESTXRAY8_BASE_DIR']
    fnames = glob.glob(os.path.join(data_dir,'images_***/images/*.png'))
    print('Number of files: {}'.format(len(fnames)))
    print('Memory used on all filenames: {:.2f} MB'.format(sys.getsizeof(fnames)/1e6))

    all_paths = []
    for fname in fnames:
        sub = fname.split('/')[-3:] 
        all_paths.append(entry_batch(path = '/'.join(sub), batch = sub[0], image_ind=sub[-1]))
    
    return all_paths

def GetBBoxList2017():
    data_dir = os.environ['CHESTXRAY8_BASE_DIR']
    meta_data = os.path.join(data_dir,'BBox_List_2017.csv')
    
    all_ents = []
    with open(meta_data, mode='r', newline='', encoding='utf-8') as ff:
        reader = csv.reader(ff)
        next(reader) # skip the header. 
        for row in reader:
            ent_ = entry_BBoxList2017(image_ind=row[0],
                                      finding_label=row[1],
                                      bboxstr=array('d', list(map(float,row[-4::]))))
            all_ents.append(ent_)
    return all_ents

def GetdataEntry2017():
    data_dir = os.environ['CHESTXRAY8_BASE_DIR']
    meta_data = os.path.join(data_dir,'Data_Entry_2017.csv')
    
    all_ents = []
    with open(meta_data, mode='r', newline='', encoding='utf-8') as ff:
        reader = csv.reader(ff)
        next(reader) # skip the header. 
        for row in reader:
            ent_ = entry_dataEntry2017(image_ind=row[0],
                                       finding_label=row[1],
                                       follow_up_num=int(row[2]),
                                       patient_id=int(row[3]),
                                       patient_age = int(row[4]),
                                       patient_gender=row[5],
                                       view_position=row[6],
                                       originalImage_wh = array('d', list(map(float,row[-4:-2]))),
                                       originalImage_xyres = array('d', list(map(float,row[-2::])))
                                       )

            all_ents.append(ent_)
    return all_ents

def startSQLLiteDB():
    # get stuff first 
    paths_ = GetImgPaths()
    print(paths_[0])
    
    bboxes = GetBBoxList2017()
    print(bboxes[0])

    da_entries = GetdataEntry2017()
    print(da_entries[0])

    # staring up the db
    dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data')
    # recursive version be careful. 
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    db_name = os.path.join(dir_, 'consolidated_sheets.db')
    con = sqlite3.connect(db_name) # can store the icd10 CM + PCS together later idk
    cur = con.cursor()
    
    # Everything from this point down could be moved into the 3 get functions for niceness. 

    cur.execute("""CREATE TABLE IF NOT EXISTS paths (
    id INTEGER PRIMARY KEY,
    image_ind TEXT,
    path TEXT,
    batch TEXT);
    """)

    for pa in paths_:
        command = """INSERT INTO paths (image_ind,path,batch) VALUES (?,?,?)"""
                # 1 indexed table
        cur.execute(command, (pa.image_ind, pa.path, pa.batch))
                              
    # not sure about the foreign key bit
    cur.execute("""CREATE TABLE IF NOT EXISTS BBoxList2017 (
    id INTEGER PRIMARY KEY,
    image_ind TEXT,
    finding_label TEXT,
    x REAL,
    y REAL,
    w REAL,
    h REAL,
    FOREIGN KEY (image_ind) REFERENCES paths(image_ind)
    );
    """)
    for bb in bboxes:
        command = """
                    INSERT INTO BBoxList2017 (image_ind, finding_label, x, y, w, h)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """
        cur.execute(command, (bb.image_ind, bb.finding_label, *bb.bboxstr))

    cur.execute("""CREATE TABLE IF NOT EXISTS dataEntry2017 (
    id INTEGER PRIMARY KEY,
    image_ind TEXT,
    finding_label TEXT,
    follow_up_num INT,
    patient_id INT, 	
    patient_age INT,
    patient_gender	TEXT,
    view_position  TEXT,
    original_w REAL,
    original_h REAL,
    original_xres REAL,
    original_yres REAL,
    FOREIGN KEY (image_ind) REFERENCES paths(image_ind)
    );
    """)

    for de in da_entries:
        command = """
                    INSERT INTO dataEntry2017 (image_ind, finding_label,
                    follow_up_num, patient_id, patient_age, patient_gender,
                    view_position,
                    original_w,
                    original_h,
                    original_xres,
                    original_yres)
                    """
        command+="VALUES ("+"?,"*11
        command=command[0:-1]+")"
        # print('Last command  ', command)
        cur.execute(command, (de.image_ind, bb.finding_label, 
                              de.follow_up_num, de.patient_id, de.patient_age, de.patient_gender,
                              de.view_position, *de.originalImage_wh, *de.originalImage_xyres))

    con.commit()# commit after insert. 
    con.close() # close .db file

def getSubsetData():
    # staring up the db
    dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data')
    db_name = os.path.join(dir_, 'consolidated_sheets.db')
  
    con = sqlite3.connect(db_name) # can store the icd10 CM + PCS together later idk
    cur = con.cursor()

    cur.execute(
        """
        SELECT
            p.path,
            b.finding_label,
            b.x, b.y, b.w, b.h,
            d.original_w, d.original_h,
            d.original_xres, d.original_yres
        FROM BBoxList2017 b
        JOIN paths p USING (image_ind)
        JOIN dataEntry2017 d USING (image_ind)
        WHERE b.finding_label LIKE ?
        """,
        ("%Effusion%",)
    )

    rows = cur.fetchall()
    for r in rows:
        print(r)

    con.close() # close .db file

if __name__ == '__main__':
    # start database - only run once. 
    # startSQLLiteDB()
    getSubsetData()
     
