__author__ = 'zuban32'

import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import nltk.tokenize as tokenize

class InsultDetector:

    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """
        # self.model = SVC(kernel='poly', degree=2)
        # self.model1 = SVC(kernel='linear', probability=True, random_state=0)
        # self.model = LogisticRegression(dual=True, multi_class='multinomial', solver='lbfgs')
        # self.model = SGDClassifier(loss='perceptron', alpha=0.000001, n_iter=50, penalty='l1')
        # self.toker = tokenize.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True, discard_empty=True)
        # self.toker = tokenize.WordPunctTokenizer()
        self.vec = CountVectorizer(tokenizer=nltk.word_tokenize, ngram_range=(1, 3))
        self.model = LinearSVC(class_weight='auto', dual=True)
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

# if __name__ == '__main__':
#     with open('discussions.json', encoding="utf-8") as data_file:
#         data = json.load(data_file)
#     dec = InsultDetector()
    # dec.extract(data)

    # X = dec.vec.fit_transform(dec.insults[0])

    # text_clf = Pipeline([('vect', CountVectorizer(tokenizer=toker.tokenize)),
    #                      ('clf', LinearSVC(C=0.5, tol=1e-5, dual=True)),
    #                      ])

    # parameters = {
    #     'vect__ngram_range': [(1, 1), (1, 2)],
        # 'clf__C': (0.5, 1.0, 1.5),
        # 'clf__tol': (1e-5, 1e-3, 1e-4),
        # 'clf__max_iter': list(range(10, 151, 10)),
        # 'clf__penalty': ('l1', 'l2'),
        # 'clf__loss': ('hinge', 'squared_hinge'),
        # 'clf__dual': (True, False)
    # }

    # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    # gs_clf = gs_clf.fit(dec.insults[0], dec.insults[1])

    # print(gs_clf.grid_scores_)

    # best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, best_parameters[param_name]))
    #
    # print('score = %d\n' % score)

    # scores = cross_validation.cross_val_score(dec.model, X, dec.insults[1], cv=10, scoring='f1', n_jobs=-1)
    # print(scores.mean())

    # dec.train(data)
    # predicted = dec.classify(data)
    # print(data == predicted)
