from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LSA():

    def __init__(self,docs):

        #converting docs to tf-idf vectors

        self.TF_IDF = TfidfVectorizer()
        self.TF_IDF.fit(docs)
        print(f" vocab : {self.TF_IDF.vocabulary_}")
        vectors = self.TF_IDF.transform(docs)

        print(f"vectors : {vectors}")
        print(f"feature_names: {self.TF_IDF.get_feature_names_out()}")
        print(f"stopwords: {self.TF_IDF.get_stop_words()}")

        #build the LSA topic model

        self.LSA_model = TruncatedSVD(n_components = 7)
        self.LSA_model.fit(vectors)
        return


    def get_features(self,new_docs):

        #get topic based features for new documents
        new_vectors = self.TF_IDF.transform(new_docs)
        return self.LSA_model.transform(new_vectors)


#initializing the LSA model
docs = ["this is a good text","here is another one"]
LSA_featurizer = LSA(docs)

#gettopic based features for the new docs
new_docs = ["this is the third text","this is the fourth text"]
LSA_features = LSA_featurizer.get_features(new_docs)
print(LSA_features)