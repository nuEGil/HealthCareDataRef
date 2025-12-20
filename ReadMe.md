# Planning 
Build this into a desktop app. 

* Update to work with the ICD codes text files under icd10cm-Code Descriptions-2026. check data directory. 

* you can do an LLM for searching local database - or do requests on a server postgres instance or whatever (simulate with pi)
swap between LLM session and CUDA image session -- can implement pytorch to do some of the image processing OR

* python subprocess.run() to have a CUDA script running in the background -- pass data back and forth instead of 
doing file I/O. 

# Coding systems
Notes on medical coding systems. Three sets of codes come up for billing and claims. ICD-10-CM, ICD-10-PCS,
CPT  

## ICD-10-CM
CDC has the ICD codes that update annually. 
https://www.cdc.gov/nchs/icd/icd-10-cm/files.html
icd10cm-Code Descriptions-2026


"" = same as above

__diagnosis codes__

3-7 characters in length 
1. Letter designates a chapter/category 
2. number forms general category
3. ""
4. alphanumeric (specificity like anatomical site or cause)
5. ""
6. ""
7. Letter or number provides qualification like type of encounter or fracture nature -- X used as a placeholder when 7th Char is required but earlier chars ar not

For a granular collection of networks you really want .

1. Letter, 26 possible values -> [0-25] 
2-3. number 100 possible values -> [0-99]
4-6. Alphanumeric - so this is 10 + 26 = 36^3 positions = 46,656 (naiive estimate) -> [0-46655] 
7. Letter or number -> 26+10 = 36 possible values [0-35]. 

Some codes are not likely  V97.33XD -- sucked into a jet engine, subsequent encounter -- low representation in data set. Some codes are invalid  -- J18.9 (Punemonia, unspecified organism). Some of these things, you need imaging to confirm. Some combinations are invalid by medical definition... 

LLMs have fixed token ranges - depending on what tokens are used in the input dictionary - 2-4 characters. check model card for this. 
GPT5 context window is 400k tokens (272k input + 128k output)
GPT5 vocabulary is unspecified but estimated at 50k tokens (on average its about 4 chars)
Tokens include non alphanumeric characters.

Output is only alphanumeric -> you can use bigrams -> 32^2 = 1296 vocabulary size on the output. 
Add a start and stop token and thats 38^2 = 1444 vocab size. so instead of a transformer with 50k vocabsize in and 50k out, you could do 50k in and 1444 out -> this gives a drastic reduction in the number of parameters. 

Split the context, input and output sequences like this
full context = Notes, Full output = V97.33XD (unlikely code)

C:\Users\gil82\progs\data\MIMIC-IV-Demo\ed

| Sample ID | Context                  | Input    | Output |
|-----------|--------------------------|----------|--------|
| 0         | Notes                    | `[Start]`| V9     |
| 1         | Notes`[Start]`           | V9       | 73     |
| 2         | Notes`[Start]`V9         | 73       | 3X     |
| 3         | Notes`[Start]`V973       | 3X       | D\0    |
| 4         | Notes`[Start]`V9733X     | D\0      | \0\0   |
| 5         | Notes`[Start]`V9733XD\0  | \0\0     | \0\0   |


Then you pick some start token, and \0\0 is the end token 
\0 is a null char. then build a tokenizer script based on this. pad the context. 
select context window length based on mean + 2 standard deviations of the max note length. 

This is a starting point for architecture design. but then the following directories have the parts necessary to build out and train this type of custom architecture. 

__See SmallNetTraining/ for a tokenizer, then a small text architecture + training script.__

__See RAGTextAppParts/hf_embed_classifiy.py for a simple example of using DistilBERT from hugging face in a classification example__

__See TransformerTutorials/ for tensorflow implementations of transformers__


### Open access Physionet data for more examples
PhysioNet has an open access data base here for more examples of ICD codes, but it's just the ICD and the corresponding name.. Realistically we would want accompanying notes that led to the diagnosis, but the triage file gives you the cheif complaint. 


#### citation for mimic iv ed demo
https://physionet.org/content/mimic-iv-ed-demo/2.2/
Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Horng, S., & Mark, R. (2023). MIMIC-IV-ED Demo (version 2.2). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/jzz5-vs76

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. 


This data set uses ICD-9 codes as a teaching example, but there should be a conversion sheet somewhere to get the ICD10 codes. 

### Cancer codes
Codes fall within the C00-C97 range for malignant neoplasms (Cancers) ICD-10-CM -- ie some codes group 
* COO - C97 : general range for all malignant neoplasms
* C00 - C75 : primary cancers of specific sites (C15-C26 for digestive organs, C50 for breast)
* C76 - C80 :  

### Doctor note formats
Another key thing. Doctors have note formats

SOAP - Gold standard -- Subjective, Objective, Assessment, Plan

BIRP - Popular in mental health, Behavior, Intervention, Response, Plan

Cheif Complaint - More concise, focusing on the main reason for the visit

## ICD-10-PCS 
__inpatient procedure codes__

* Exactly 7 alphanumeric characters
* each character is an axis of classification, representing specific details about the procedure  (body part, approach, device used)

ok. so this one is easier it has 

## CPT 
__outpatient procedures/services__

* generally 5 numeric characters
* may require addition of 2 character modifiers to provide extra information about the procedure. 


So say you have some amount of data, images, patient notes, doctors notes, etc. Your job is to find a transform that takes that data as an input, then outputs the text code of interest.... 

# Database search tool 
Find a code .com --> huge knowledge base for this stuff. you can enter decriptions, the ICD code, symptoms

https://www.findacode.com/icd-10-cm

CMS.gov  has the present on admission codes (POA)
https://www.cms.gov/medicare/payment/fee-for-service-providers/hospital-aquired-conditions-hac/coding

Look into UB-04/837 claim structure 
encounter_id
patient_id
diagnosis_seq_number
diagnosis_code
diagnosis_code_system (ICD-10-CM vs ICD-9)
diagnosis_description
diagnosis_type
poa_indicator
poa_required_flag
diagnosis_date

## findacode.com notes
 findacode.com  has the documentation + the Dorlands illustrated dictionary. subscribers can add their own notes

Find a code also has a test name
acquired hemolyptic anemia -- ICD-10-CM codes and diagnostic testing/screening initial testing...

## thoughts on GPT

GPT 5.2 has the language model + internet search feature.  

wait think. 
1. User input into LLM -- keywords like find, search , and then a specific thing -- constraints -- ICD, CPT, inpatient vs outpatient,.. 

2. have a routing model parse the text and figure out what it needs to do.... 
	-- answer directly, 
	-- search required
	-- calculator
	-- medical lookup


3. if it's a search -- call something like Google API for programmable search or you might even have your own web crawler... 
	-- Google Programmable search 
	-- BingAPI
	-- curated sources like CMS tables, ICD sites
	-- internal DB
	-- controlled crawler 

4. ok you get back successful search results --> condense them down with llm summary 

5. format links, summary and response text. 

some other decision making things like whether its legal/medical/financial

The LLM parses text, and you can build a lot of other systems around that text parsing with tools that already exist... caveat, is how much the API calls cost... Right because at that point GPT can only do as much as its API calls + the LLM, then the income stream in is user subscriptions, business to business deals, and price per token from the Open AI API


Healthcare and Clinical data companies already are database companies with a financial service attached. It's better if the AI companied handle training the models, and building out tools around the models. -- because they have all the data.. 
