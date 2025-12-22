# Text input cases to handle
1. single key word
2. multiple space separated key words
3. special : single -- so example would be "header: 09A" --> like if someone already knows that header from working with these 
4. special : multiple  -- "header : 09A M61 T754 " --> posining, burn, electrocution  
5. LLM : single / multipl
6. LLM : header : single / multiple

Consider using button or check boxes --> user doesnt have to memorize formatting, and the ui can append these prefixes. 

On the LLM side - you can use a prompt that says to generate key words given this querry --> so text = generator()
then do  
keywords = list(set(text.strip(special chars).split()))
then key words is now in the same logic path as the base keyword based SQL querries. boom. llm text normalized inputs. 

Stretch --> finetune even gemma3 and gpt2 on long_desc parts --> there should be enough data within the CDC website files to 
do that -- if llm poinsoning can be accomplished in only a couple 100 documents, you can totally fine tune here --> those 
db text files are like 14MB worth of text a pop. 14 million chars per text file -- crime and punishment is 1.1 mil chars
so you get 14 crime and punishments every FY2X database (FY26, FY25, FY24...) so on. 

# update to include codes from other databases 
1. Procedure Coding System 
2. Various years of ICD CM and ICD PCS