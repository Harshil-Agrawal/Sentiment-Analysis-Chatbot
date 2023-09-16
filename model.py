# Libraries


import re
import string
import spacy
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter
import warnings
nltk.download("vader_lexicon")
nltk.download('stopwords')
warnings.filterwarnings('ignore')

# import en_core_web_sm
# nlp = en_core_web_sm.load()

# Model

class PredictReview:
    
    def vectorize(self,train_data,test_data):
        
        tfidf = TfidfVectorizer()
        train = tfidf.fit_transform(train_data.values.astype('U'))
        test = tfidf.transform(test_data.values.astype('U'))
        
        return train,test,tfidf
    
    def split(self,data,train_size=0.1,shuffle=101):
        
        input_data = data['review']
        output_data = data['label_num']
        train_data, test_data, train_output, test_output = train_test_split(input_data, output_data, test_size=train_size, random_state=shuffle)
        return train_data, test_data, train_output, test_output
    
    def base(self,data):
        
        log_reg = LogisticRegression()
        data = self.prepare_data_for_train(data)
        train_data, test_data, train_output, test_output = self.split(data)
        train,test,tfidf = self.vectorize(train_data,test_data)
        log_reg.fit(train,train_output)
        pred = log_reg.predict(test)
#         self.performance(pred,test,test_output,log_reg)
        return log_reg,tfidf
    
   
    def prepare_data_for_train(self,input_data):
        
        nlp = spacy.load('en_core_web_sm')
        stopword = nltk.corpus.stopwords.words('english')
        empty_list  = []
        for text in input_data.review:
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = re.sub(r'[^\w\s]', '',str(text))            
            text=re.split("\W+",text)                          
            text=[word for word in text if word not in stopword]
            text = ' '.join(text)       
            empty_list.append(text)
        #input_data.drop('review',axis=1,inplace=True)
        input_data['review'] = empty_list
        return input_data
    
        
    def test_sample(self,text,tfidf,base_model):
        
        text = self.clean_df(text)
        text_sample = tfidf.transform([text])
        pred = base_model.predict(text_sample)
        if pred[0] == 1:
            return 'Positive 🙂'
        else:
            return 'Negative 🙁'
        
    def  clean_df(self,text):
        text = text.lower()
        nlp = spacy.load('en_core_web_sm')
        stopword = nltk.corpus.stopwords.words('english')
        text = re.sub(r'[^\w\s]', '',str(text))            
        text=re.split("\W+",text)                          
        text=[word for word in text if word not in stopword]
        text = ' '.join(text)                              
        return text