import spacy
import multiprocessing

spacy_nlp = spacy.load('en')

def test(i):    
    txt_1st = "this is the first sentence of {}".format(i)
    txt_2nd = "this is the second sentence of {}".format(i)
    
    print(txt_1st)
    parsed_txt_1st = spacy_nlp(txt_1st)
    print(txt_2nd)
    parsed_txt_2nd = spacy_nlp(txt_2nd)
    return parsed_txt_1st, parsed_txt_2nd


pool = multiprocessing.Pool(processes=3)
data = range(3)
results = pool.map(test, data)
pool.close()
pool.join()
    
print(results)