from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDA:
    def __init__(self, docs):
        # convert docs to TFIDF vectors
        self.TF = CountVectorizer()
        self.TF.fit(docs)
        vectors = self.TF.transform(docs)

        # build the LDA model

        self.LDA_model = LatentDirichletAllocation(n_components=50)
        self.LDA_model.fit(vectors)

        return

    def get_features(self, new_docs):
        # topic based features

        new_vectors = self.TF.transform(new_docs)
        return self.LDA_model.transform(new_vectors)


# initializing the LSA model
docs = ["this is a good text", "here is another one"]
LDA_featurizer = LDA(docs)

# gettopic based features for the new docs
new_docs = ["this is the third text", "this is the fourth text"]
LDA_features = LDA_featurizer.get_features(new_docs)
print(LDA_features)
