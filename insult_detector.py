__author__ = 'zuban32'

import json
import nltk
import numpy
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

class InsultDetector:

    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """
        # self.model = SVC(kernel='poly', degree=2)
        # self.model1 = SVC(kernel='linear', probability=True, random_state=0)
        self.model1 = LogisticRegression(dual=True, multi_class='multinomial', solver='lbfgs')
        self.model2 = SGDClassifier(loss='perceptron', alpha=0.000001, n_iter=50, penalty='l1')
        # self.model1 = LinearSVC()
        self.insults = [list(), list()]
        self.results = [list(), list()]
        self.num = 0

    def extract_post(self, post):
        if 'insult' in post.keys():
            self.insults[0].append(post['text'])
            self.insults[1].append(post['insult'])
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
        # max_len = 0
        # for post in self.insults[0]:
        #     cur_len = len(nltk.word_tokenize(post))
        #     if cur_len > max_len:
        #         max_len = cur_len
        self.vec = CountVectorizer()
        X = self.vec.fit_transform(self.insults[0])
        self.model1.fit(X, self.insults[1])
        self.model2.fit(X, self.insults[1])

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
        self.results[1] = list(map(lambda x, y: x or y, self.model1.predict(X), self.model2.predict(X)))

        for discussion in unlabeled_discussions:
            root = discussion['root']
            if root['children']:
                for post in root['children']:
                    self.set_result(post)

        return unlabeled_discussions

# if __name__ == '__main__':
#     with open('discussions.json', encoding="utf-8") as data_file:
#         data = json.load(data_file)
#     dec = InsultDetector()
#     dec.extract(data)

    # max_len = 0
    # sum_len = 0
    # for post in dec.insults[0]:
    #     cur_len = len(nltk.word_tokenize(post))
    #     sum_len += cur_len
    #     if cur_len > max_len:
    #         max_len = cur_len
    # sum_len /= len(dec.insults[0])
    # sum_len = int(sum_len)

    # cv_folds = 10
    #
    # dec.vec = CountVectorizer(ngram_range=(1, 1))
    # X = dec.vec.fit_transform(dec.insults[0])
    # X = normalize(X.)
    # part_len = int(len(dec.insults[0]) / cv_folds)
    # new_data = numpy.array(dec.insults[0], copy=False)
    # new_res = numpy.array(dec.insults[1], copy=False)
    # total = 0
    #
    # for i in range(cv_folds):
    #     mask_train = numpy.ones(new_data.shape, dtype=bool)
    #     mask_pred = numpy.zeros(new_res.shape, dtype=bool)
    #     for j in range(i * part_len, (i + 1) * part_len):
    #         mask_train[j] = 0
    #         mask_pred[j] = 1
    #     X_i = dec.vec.fit_transform(new_data[mask_train])
    #     dec.model1.fit(X_i, new_res[mask_train])
    #     dec.model2.fit(X_i, new_res[mask_train])
    #     dec.model3.fit(X_i, new_res[mask_train])
    #     Y_i = dec.vec.transform(new_data[mask_pred])
    #     res1 = dec.model1.predict(Y_i)
    #     res2 = dec.model2.predict(Y_i)
    #     res3 = dec.model3.predict(Y_i)
    #     res = list(map(lambda x,y,z: (x and y) or (y and z) or (x and z), res1, res2, res3))
    #     # print('i = %d, res = ' % i)
    #     # print(res)
    #     # print('\n')
    #     # for post, true, pred in zip(new_data[mask_pred], new_res[mask_pred], res):
    #     #     if true and not pred:
    #     #         print('FP: %s\n' + post)
    #     #     elif pred and not true:
    #     #         print('Missed: $s\n' + post)
    #     total += f1_score(new_res[mask_pred], res)
    #     # print('i = %d, score = %f\n' % (i, f1_score(new_res[mask_pred], res)))

    # scores = cross_validation.cross_val_score(dec.model, X, dec.insults[1], cv=5, scoring='f1', n_jobs=4)
    # print(scores.mean())
#
    # dec.train(data)
    # predicted = dec.classify(data)
    # print(data == predicted)