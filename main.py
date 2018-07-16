import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import os.path
import nlp_tasks
from sklearn.cross_validation import cross_val_score

from sklearn.neural_network import MLPClassifier  # アルゴリズムとしてmlpを使用
def train():
    classifier = MyMLPClassifier()
    classifier.train('corpus.csv')
    # classifier = MyMLPClassifier()
    # classifier.cross_validation('corpus.csv')

def predict():
    classifier = MyMLPClassifier()
    classifier.load_model()
    result = classifier.predict("０日正午頃、前橋市岩神町の「ベイシアスーパーマーケット前橋岩神店」で、店員が男に刺されたと１１０番があった。店員の男女２人がけがを負い、救急搬送された。群馬県警は、包丁２本を持っていた男を殺人未遂の疑いで現行犯逮捕し、身元などを調べている。現場は群馬県庁から北に約１・３キロの住宅街。")
    print(result)



class MyMLPClassifier():
    model = None
    model_name = "mlp"

    def load_model(self):
        if os.path.exists(self.get_model_path())==False:
            raise Exception('no model file found!')
        self.model = joblib.load(self.get_model_path())
        self.classes =  joblib.load(self.get_model_path('class')).tolist()
        self.vectorizer = joblib.load(self.get_model_path('vect'))
        self.le = joblib.load(self.get_model_path('le'))

    def get_model_path(self,type='model'):
        return 'models/'+self.model_name+"_"+type+'.pkl'#提前建一个models文件夹

    def get_vector(self,text):
        return self.vectorizer.transform([text])



    def train(self, csvfile):
        df = pd.read_csv(csvfile,names=('text','category'))
        X, vectorizer = nlp_tasks.get_vector_by_text_list(df["text"])

        le = LabelEncoder()
        le.fit(df['category'])
        Y = le.transform(df['category'])

        model = MLPClassifier(max_iter=300, hidden_layer_sizes=(100,),verbose=10,)
        model.fit(X, Y)

        # save models
        joblib.dump(model, self.get_model_path())
        joblib.dump(le.classes_, self.get_model_path("class"))
        joblib.dump(vectorizer, self.get_model_path("vect"))
        joblib.dump(le, self.get_model_path("le"))

        self.model = model
        self.classes = le.classes_.tolist()
        self.vectorizer = vectorizer

    def predict(self,query):
        X = self.vectorizer.transform([query])
        key = self.model.predict(X)
        return self.classes[key[0]]


    def cross_validation(self, csvfile):
        self.model = MLPClassifier(max_iter=300, hidden_layer_sizes=(100,), verbose=10, )
        df = pd.read_csv(csvfile, names=('text', 'category'))
        _items = df["text"]
        X, vectorizer = nlp_tasks.get_vector_by_text_list(_items)

        # loading labels
        le = LabelEncoder()
        le.fit(df['category'])
        Y = le.transform(df['category'])

        model = MLPClassifier(max_iter=300, hidden_layer_sizes=(100,), verbose=10, )
        model.fit(X, Y)

        # save models
        joblib.dump(model, self.get_model_path())
        joblib.dump(le.classes_, self.get_model_path("class"))
        joblib.dump(vectorizer, self.get_model_path("vect"))
        joblib.dump(le, self.get_model_path("le"))

        self.classes = le.classes_.tolist()
        self.vectorizer = vectorizer
        scores = cross_val_score(self.model, X, Y, cv=4)
        print(scores)
        print(np.average(scores))


if __name__ == '__main__':
    # train()
    predict()

