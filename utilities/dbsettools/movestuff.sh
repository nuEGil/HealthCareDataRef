# mover shell script if you decide to hold the crawl results in one directory during the run and need to move to your long term storage. 
mv path/to/new_crawl/google_results/*.json path/to/base_crawl/google_results/
mv path/to/new_crawl/wikipedia_results/*.html path/to/base_crawl/wikipedia_results/

# could do it with rsync. and then delete from the source dir. 
