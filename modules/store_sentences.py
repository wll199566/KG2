"""
This script is for storing all the sentences in ARC_Corpus_clean.txt into the elasticsearch database.
"""

from elasticsearch import Elasticsearch

def store_sentences(input_file):
    """
    Arguments:
            input_file: path to ARC_Corpus_clean.txt        
    """
    
    es = Elasticsearch()
 
    with open(input_file, 'rt') as fin:
        for i, line in enumerate(fin, 1):
            sentence = {
                "text": line
            }
            resp = es.index(index="arc_corpus_clean", doc_type="sentences", id=i, body=sentence)
            if i%10000 == 0:
                print(i, " sentences has been stored!!")
    print("Finish!!")            

if __name__ == "__main__":
    
    input_file = "../data/ARC_Corpus_clean.txt"
    store_sentences(input_file)