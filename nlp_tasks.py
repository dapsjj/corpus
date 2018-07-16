import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def _split_to_words(text):
    tagger = MeCab.Tagger('-Owakati')
    try:
        res = tagger.parse(text.strip())
    except:
        return []
    return res



def get_vector_by_text_list(_items):
    # count_vect = CountVectorizer(analyzer=_split_to_words)
    count_vect = TfidfVectorizer(analyzer=_split_to_words)
    bow = count_vect.fit_transform(_items)
    X = bow.todense()
    return [X,count_vect]
