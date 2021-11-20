from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class TccSvmMetrics:

    def __init__(self, y_test = None, y_pred = None, tp = 0, fn = 0, fp = 0, tn = 0):
        self.__y_test = y_test
        self.__y_pred = y_pred
        self.__tp = tp
        self.__fn = fn
        self.__fp = fp
        self.__tn = tn

        self.__accuracy_score = 0
        self.__f1_score = 0
        self.__precision_score = 0
        self.__recall_score = 0
        self.__sensitivity = 0
        self.__specificity = 0

    @property
    def accuracy_score(self):
        return self.__accuracy_score

    @accuracy_score.setter
    def accuracy_score(self, accuracy_score):
        self.__accuracy_score = accuracy_score

    @property
    def f1_score(self):
        return self.__f1_score

    @f1_score.setter
    def f1_score(self,f1_score):
        self.__f1_score = f1_score

    @property
    def precision_score(self):
        return self.__precision_score

    @precision_score.setter
    def precision_score(self,precision_score):
        self.__precision_score = precision_score

    @property
    def recall_score(self):
        return self.__recall_score

    @recall_score.setter
    def recall_score(self,recall_score):
        self.__recall_score = recall_score

    @property
    def sensitivity(self):
        return self.__sensitivity

    @sensitivity.setter
    def sensitivity(self,sensitivity):
        self.__sensitivity = sensitivity

    @property
    def specificity(self):
        return self.__specificity

    @specificity.setter
    def specificity(self,specificity):
        self.__specificity = specificity

    @property
    def roc_auc_score(self):
        return self.__roc_auc_score

    @roc_auc_score.setter
    def roc_auc_score(self, roc_auc_score):
        self.__roc_auc_score = roc_auc_score

    def executar_metrics(self):
        self.__accuracy_score = accuracy_score(self.__y_test, self.__y_pred)
        self.__f1_score = f1_score(self.__y_test, self.__y_pred)
        self.__precision_score = precision_score(self.__y_test, self.__y_pred)
        self.__recall_score = recall_score(self.__y_test, self.__y_pred)
        self.__sensitivity = self.__tp / (self.__tp + self.__fn)
        self.__specificity = self.__tn / (self.__fp + self.__tn)
