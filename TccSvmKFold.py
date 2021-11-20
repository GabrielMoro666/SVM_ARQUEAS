import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from TccRocCurve import TccRocCurve
from TccSvmMetrics import TccSvmMetrics


class TccSvmKFold:

    def __init__(self, classe, x, y, train_indices, test_indices, kernel, decision_limite):
        self.__classe = classe

        self.__x = x
        self.__y = y

        self.__train_indices = train_indices
        self.__test_indices = test_indices

        self.__kernel = kernel

        self.__decision_limite = decision_limite

        self.__x_train = None
        self.__y_train = None
        self.__x_test = None
        self.__x_test = None
        self.__svc = None
        self.__y_pred = None
        self.__y_pred_proba = None
        self.__y_pred_no_proba = None
        self.__tp = 0
        self.__fn = 0
        self.__fp = 0
        self.__tn = 0
        self.__svm_metrics = None
        self.__roc_curve = None

        self.__x_test = x.iloc[test_indices]
        self.__y_test = y.iloc[test_indices]

        self.__x_train = x.iloc[train_indices]
        self.__y_train = y.iloc[train_indices]

    @property
    def train_indices(self):
        return self.__train_indices

    @train_indices.setter
    def train_indices(self, train_indices):
        self.__train_indices = train_indices

    @property
    def test_indices(self):
        return self.__test_indices

    @test_indices.setter
    def test_indices(self, test_indices):
        self.__test_indices = test_indices

    @property
    def x_train(self):
        return self.__x_train

    @x_train.setter
    def x_train(self, x_train):
        self.__x_train = x_train

    @property
    def y_train(self):
        return self.__y_train

    @y_train.setter
    def y_train(self, y_train):
        self.__y_train = y_train

    @property
    def x_test(self):
        return self.__x_test

    @x_test.setter
    def x_test(self, x_test):
        self.__x_test = x_test

    @property
    def y_test(self):
        return self.__y_test

    @y_test.setter
    def y_test(self, y_test):
        self.__y_test = y_test

    @property
    def svc(self):
        return self.__svc

    @svc.setter
    def svc(self, svc):
        self.__svc = svc

    @property
    def y_pred(self):
        return self.__y_pred

    @y_pred.setter
    def y_pred(self, y_pred):
        self.__y_pred = y_pred

    @property
    def y_pred_proba(self):
        return self.__y_pred_proba

    @y_pred_proba.setter
    def y_pred_proba(self, y_pred_proba):
        self.__y_pred_proba = y_pred_proba

    @property
    def y_pred_no_proba(self):
        return self.__y_pred_no_proba

    @y_pred_no_proba.setter
    def y_pred_no_proba(self, y_pred_no_proba):
        self.__y_pred_no_proba = y_pred_no_proba

    @property
    def tp(self):
        return self.__tp

    @tp.setter
    def tp(self, tp):
        self.__tp = tp

    @property
    def fn(self):
        return self.__fn

    @fn.setter
    def fn(self, fn):
        self.__fn = fn

    @property
    def fp(self):
        return self.__fp

    @fp.setter
    def fp(self, fp):
        self.__fp = fp

    @property
    def tn(self):
        return self.__tn

    @tn.setter
    def tn(self, tn):
        self.__tn = tn

    @property
    def svm_metrics(self):
        return self.__svm_metrics

    @svm_metrics.setter
    def svm_metrics(self, svm_metrics):
        self.__svm_metrics = svm_metrics

    @property
    def roc_curve(self):
        return self.__roc_curve

    @roc_curve.setter
    def roc_curve(self, roc_curve):
        self.__roc_curve = roc_curve

    def treinar_svm(self):
        random_state = np.random.RandomState(0)

        self.__svc = SVC(kernel=self.__kernel, C=1, probability=True, random_state=random_state)
        self.__svc.fit(self.__x_train, self.__y_train)

    def predicao_svm(self):
        self.__y_pred_proba = self.__svc.predict_proba(self.__x_test)
        self.__y_pred = self.__y_pred_proba[:, 1] >= self.__decision_limite
        self.__y_pred_no_proba = [0 for _ in range(len(self.__x_test))]

    def executar_confusion_matrix(self):
        self.__tp, self.__fn, self.__fp, self.__tn = confusion_matrix(self.__y_test, self.__y_pred).ravel()

    def executar_metrics(self):
        self.__svm_metrics = TccSvmMetrics(y_test = self.__y_test, y_pred = self.__y_pred, tp = self.__tp, fn = self.__fn, fp = self.__fp, tn = self.__tn)
        self.__svm_metrics.executar_metrics()

    def executar_roc_curve(self):
        self.__roc_curve = TccRocCurve(classe=self.__classe, y_test=self.__y_test, y_pred=self.__y_pred, y_pred_proba=self.__y_pred_proba)
        self.__roc_curve.executar_roc_curve()
