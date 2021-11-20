
class TccCurve:

    def __init__(self, label, fpr, tpr):
        self.__fpr = fpr
        self.__tpr = tpr
        self.__label = label

    @property
    def fpr(self):
        return self.__fpr

    @fpr.setter
    def fpr(self, fpr):
        self.__fpr = fpr

    @property
    def tpr(self):
        return self.__tpr

    @tpr.setter
    def tpr(self, tpr):
        self.__tpr = tpr

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self,label):
        self.__label = label