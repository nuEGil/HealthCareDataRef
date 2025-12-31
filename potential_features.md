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

# notes on threading...
in some cases someone enters multiple key words -- you want it to search for each word in parallel unless its "" in which case you search for the whole thing together. Then you rank though?


# Special commands
not implemented just looking
search engine optimization makes use of these

## Google
1. "find the exact text in this order"
2. (-) minus excludes words -- so like jaguar speed -car
3. * wild card for unknown words -- a * saved is a * earned
4. sites: searches a specific website -->    site:wikipedia.org "moon landing"
5. filetype: finds specific file types -- filetype:pdf
6. (Two dots) .. searches a ranges of numbers, dates --> camera $200 .. $500
7. OR capitalized: finds results with either term
8. @ symbol finds social media handles.... 
9. Google has some code specific things..  C++, ==, +=, now recognizes. 

## DuckDuckGo
1. bangs for instant site saerch !w[term] wikipedia. !a[term]:Amazon. !yt[term] youtube
2. "" exact phrase
3. + prioritize a word
4. site:domain.com
5. filetype:pdf
6. intitle:term --> find pages with the term in the title. 
7. inurl:term --> find pages with the term in the url 
8. \\ Im feeling ducky.... takes you straight to the first result so \\fun facts
9. @ social media pages
10. #definitions  so like #definition apoplegic
11. $ search for stock prices --> $SPY, $BTC

# update to include codes from other databases 
1. Procedure Coding System 
2. Various years of ICD CM and ICD PCS