__author__ = 'zuban32'

import json
import nltk
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import nltk.tokenize as tokenize

class InsultDetector:

    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """
        # self.model = SVC(kernel='poly', degree=2, class_weight='auto')
        # self.model1 = SVC(kernel='linear', probability=True, random_state=0)
        self.model = LogisticRegression(class_weight='auto')
        # self.model = SGDClassifier(class_weight='auto', loss='log', alpha=0.000001, n_iter=50)
        # self.toker = tokenize.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True, discard_empty=True)
        # self.toker = tokenize.WordPunctTokenizer()
        self.vec = CountVectorizer(tokenizer=nltk.word_tokenize, ngram_range=(1, 2))
        # self.model = LinearSVC(class_weight='auto', dual=False)
        self.insults = [list(), list()]
        self.results = [list(), list()]
        self.num = 0

    def extract_post(self, post):
        if 'insult' in post.keys():
            self.insults[0].append(post['text'])
            self.insults[1].append(post['insult'])
            # if post['insult']:
            #     print(post['text'] + '\n' + str(post['insult']))
        if 'children' in post.keys():
            for post in post['children']:
                self.extract_post(post)

    def extract(self, discussions):
        for discussion in discussions:
            root = discussion['root']
            if root['children']:
                for post in root['children']:
                    self.extract_post(post)

    def train(self, labeled_discussions):
        """
        This method train the model.
        :param discussions: the list of discussions. See description of the discussion in the manual.
        :return: None
        """
        self.extract(labeled_discussions)
        X = self.vec.fit_transform(self.insults[0])
        self.model.fit(X, self.insults[1])

    def classify_post(self, post):
        if 'insult' in post.keys():
            self.results[0].append(post['text'])
        if 'children' in post.keys():
            for post in post['children']:
                self.classify_post(post)

    def set_result(self, post):
        if 'insult' in post.keys():
            post['insult'] = self.results[1][self.num]
            self.num += 1
        if 'children' in post.keys():
            for post in post['children']:
                self.set_result(post)

    def classify(self, unlabeled_discussions):
        """
        This method take the list of discussions as input. You should predict for every message in every
        discussion (except root) if the message insult or not. Than you should replace the value of field "insult"
        for True if the method is insult and False otherwise.
        :param discussion: list of discussion. The field insult would be replaced by False.
        :return: None
        """
        for discussion in unlabeled_discussions:
            root = discussion['root']
            if root['children']:
                for post in root['children']:
                    self.classify_post(post)
        X = self.vec.transform(self.results[0])
        self.results[1] = self.model.predict(X)

        for discussion in unlabeled_discussions:
            root = discussion['root']
            if root['children']:
                for post in root['children']:
                    self.set_result(post)

        return unlabeled_discussions

if __name__ == '__main__':
    with open('discussions.json', encoding="utf-8") as data_file:
        data = json.load(data_file)
    dec = InsultDetector()
    dec.extract(data)

    X = dec.vec.fit_transform(dec.insults[0])

    # n_folds = 5
    # kf = cross_validation.KFold(len(dec.insults[0]), n_folds=n_folds)
    # data = numpy.array(dec.insults[0])
    # res = numpy.array(dec.insults[1])
    #
    # scores = numpy.zeros(n_folds)
    # i = 0
    #
    # for train_index, test_index in kf:
    #     model = LinearSVC(class_weight='auto', dual=True)
    #
    #     X = dec.vec.fit_transform(data[train_index])
    #     model.fit(X, res[train_index])
    #     X = dec.vec.transform(data[test_index])
    #     cur_res = model.predict(X)
    #
    #     # for post, real, cur in zip(data[test_index], res, cur_res):
    #     #     if real and not cur:
    #     #         print('Missed\n' + post)
    #     #     elif not real and cur:
    #     #         print('FP\n' + post)
    #
    #     scores[i] = f1_score(res[test_index], cur_res)
    #     print('Score[%d] = %f\n' % (i, scores[i]))
    #     i += 1
    #
    # print(scores.mean())

    scores = cross_validation.cross_val_score(dec.model, X, dec.insults[1], cv=10, scoring='f1', n_jobs=-1)
    print(scores.mean())

    # dec.train(data)
    # predicted = dec.classify(data)
    # print(data == predicted)
