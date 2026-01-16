#!/bin/bash
# Order to run these srcipts. dont have to run all 
set -e

python parseICD10.py
python parseMimic.py
python save_wikipedia_pages.py
python parse_wikipedia_pages.py
# need to write this one. 
# python enrich_icd10_with_symptoms.py