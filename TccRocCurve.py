from matplotlib import pyplot
from sklearn.metrics import roc_curve, roc_auc_score

from TccCurve import TccCurve

class TccRocCurve:

    def __init__(self, classe, y_test=None, y_pred=None, y_pred_proba=None, fpr=[], tpr=[]):
        self.__classe = classe
        self.__y_test = y_test
        self.__y_pred = y_pred
        self.__y_pred_proba = y_pred_proba

        self.__fpr = fpr
        self.__tpr = tpr
        self.__roc_auc_score = []
        self.__roc_auc_score_average = 0
        self.__curvas = list()

    def add_curva(self, label, fpr, tpr):
        self.__curvas.append(TccCurve(label,fpr,tpr))

    @property
    def classe(self):
        return self.__classe

    @classe.setter
    def classe(self, classe):
        self.__classe = classe

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
    def roc_auc_score(self):
        return self.__roc_auc_score

    @roc_auc_score.setter
    def roc_auc_score(self, roc_auc_score):
        self.__roc_auc_score = roc_auc_score

    @property
    def roc_auc_score_average(self):
        return self.__roc_auc_score_average

    @roc_auc_score_average.setter
    def roc_auc_score_average(self, roc_auc_score_average):
        self.__roc_auc_score_average = roc_auc_score_average

    def ordenar_fpr_tpr(self, fpr=[], tpr=[]):
        if len(fpr) == 0:
            fpr = self.__fpr
        if len(tpr) == 0:
            tpr = self.__tpr

        ordenado = False

        while not ordenado:
            ordenado = True

            for i in range(len(fpr) - 1):
                if fpr[i] > fpr[i + 1]:
                    tpr[i], tpr[i + 1] = tpr[i + 1], tpr[i]
                    fpr[i], fpr[i + 1] = fpr[i + 1], fpr[i]

                    ordenado = False
                elif fpr[i] == fpr[i + 1]:
                    if tpr[i] > tpr[i + 1]:
                        tpr[i], tpr[i + 1] = tpr[i + 1], tpr[i]
                        fpr[i], fpr[i + 1] = fpr[i + 1], fpr[i]

                        ordenado = False

        return fpr, tpr

    def executar_roc_curve(self):
        self.__fpr = []
        self.__tpr = []
        self.__roc_auc_score = []

        self.__fpr.append(0.)
        self.__tpr.append(0.)

        for indice_decision_limit in range(1, 10):
            decision_limit = (indice_decision_limit / 10)

            self.__roc_auc_score.append(roc_auc_score(self.__y_test, self.__y_pred_proba[:, 1] >= decision_limit))

            fpr, tpr, _ = roc_curve(self.__y_test, self.__y_pred_proba[:, 1] >= decision_limit, pos_label=1)

            self.__fpr.append(fpr[1])
            self.__tpr.append(tpr[1])

        self.__fpr.append(1.)
        self.__tpr.append(1.)

        self.__fpr, self.__tpr = self.ordenar_fpr_tpr(self.__fpr, self.__tpr)

        self.calcular_roc_auc_score_average()

    def plotar_roc_curve(self, multicurvas = False, nome_grafico = "", exportar = False, diretorio = "", nome_arquivo = ""):
        fpr_no_proba = [0., 1.]
        tpr_no_proba = [0., 1.]

        pyplot.title(nome_grafico)

        pyplot.plot(fpr_no_proba, tpr_no_proba, linestyle='--')

        if not multicurvas:
            pyplot.plot(self.__fpr, self.__tpr, marker='.', label=self.__classe.upper())
        else:
            for c in self.__curvas:
                pyplot.plot(c.fpr, c.tpr, marker='.', label=c.label.upper())

        pyplot.xlabel('False positive rate')
        pyplot.ylabel('True positive rate')

        pyplot.legend()

        if not exportar:
            pyplot.show()
            pyplot.clf()
        else:
            pyplot.savefig(diretorio + "/" + nome_arquivo, dpi=600)
            pyplot.clf()

    def calcular_roc_auc_score_average(self):
        soma = 0
        qtde = 0

        for rca in self.__roc_auc_score:
            qtde += 1
            soma += rca

        self.__roc_auc_score_average = soma / qtde
