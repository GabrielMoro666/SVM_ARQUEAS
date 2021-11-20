import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from TccRocCurve import TccRocCurve
from TccSvmKFold import TccSvmKFold
from TccSvmMetrics import TccSvmMetrics


class TccSvm:

    def __init__(self, arquivo, classe, posicao_classe):
        self.__arquivo = arquivo
        self.__classe = classe
        self.__posicao_classe = posicao_classe

        self.__kernel = ""
        self.__k_fold = 0
        self.__decision_limite = 0.5
        self.__dados = None
        self.__x = None
        self.__y = None
        self.__k_fold_lista = None
        self.__svm_metrics_average = None
        self.__fpr_average = []
        self.__tpr_average = []
        self.__roc_auc_score_average = 0
        self.__roc_curve_average = None
        self.__tp_average = 0
        self.__fn_average = 0
        self.__fp_average = 0
        self.__tn_average = 0

        self.read_csv_dados()

    @property
    def classe(self):
        return self.__classe

    @classe.setter
    def classe(self, classe):
        self.__classe = classe

    @property
    def kernel(self):
        return self.__kernel

    @kernel.setter
    def kernel(self, kernel):
        self.__kernel = kernel

    @property
    def decision_limite(self):
        return self.__decision_limite

    @decision_limite.setter
    def decision_limite(self, decision_limite):
        self.__decision_limite = decision_limite

    @property
    def k_fold(self):
        return self.__k_fold

    @k_fold.setter
    def k_fold(self, k_fold):
        self.__k_fold = k_fold

    @property
    def dados(self):
        return self.__dados

    @dados.setter
    def dados(self, dados):
        self.__dados = dados

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    @property
    def k_fold_lista(self):
        return self.__k_fold_lista

    @k_fold_lista.setter
    def k_fold_lista(self, k_fold_lista):
        self.__k_fold_lista = k_fold_lista

    @property
    def svm_metrics_average(self):
        return self.__svm_metrics_average

    @svm_metrics_average.setter
    def svm_metrics_average(self, svm_metrics_average):
        self.__svm_metrics_average = svm_metrics_average

    @property
    def fpr_average(self):
        return self.__fpr_average

    @fpr_average.setter
    def fpr_average(self, fpr_average):
        self.__fpr_average = fpr_average

    @property
    def tpr_average(self):
        return self.__tpr_average

    @tpr_average.setter
    def tpr_average(self, tpr_average):
        self.__tpr_average = tpr_average

    @property
    def roc_auc_score_average(self):
        return self.__roc_auc_score_average

    @roc_auc_score_average.setter
    def roc_auc_score_average(self, roc_auc_score_average):
        self.__roc_auc_score_average = roc_auc_score_average

    @property
    def roc_curve_average(self):
        return self.__roc_curve_average

    @roc_curve_average.setter
    def roc_curve_average(self, roc_curve_average):
        self.__roc_curve_average = roc_curve_average

    @property
    def tp_average(self):
        return self.__tp_average

    @tp_average.setter
    def tp_average(self, tp_average):
        self.__tp_average = tp_average

    @property
    def fn_average(self):
        return self.__fn_average

    @fn_average.setter
    def fn_average(self, fn_average):
        self.__fn_average = fn_average

    @property
    def fp_average(self):
        return self.__fp_average

    @fp_average.setter
    def fp_average(self, fp_average):
        self.__fp_average = fp_average

    @property
    def tn_average(self):
        return self.__tn_average

    @tn_average.setter
    def tn_average(self, tn_average):
        self.__tn_average = tn_average

    def read_csv_dados(self):
        self.__dados = pd.read_csv(self.__arquivo, sep=";")

        self.__x = self.__dados.drop(self.__posicao_classe, axis=1)
        self.__y = self.__dados[self.__posicao_classe]

    def executar_k_fold(self):
        r_k_fold = RepeatedKFold(n_splits=self.__k_fold, n_repeats=1)
        self.__k_fold_lista = []

        for train_indices, test_indices in r_k_fold.split(self.__x):
            tcc_svm_k_fold_k = TccSvmKFold(self.__classe, self.__x, self.__y, train_indices, test_indices,
                                           self.__kernel, self.__decision_limite)
            tcc_svm_k_fold_k.treinar_svm()
            tcc_svm_k_fold_k.predicao_svm()
            tcc_svm_k_fold_k.executar_confusion_matrix()
            tcc_svm_k_fold_k.executar_metrics()
            tcc_svm_k_fold_k.executar_roc_curve()

            self.__k_fold_lista.append(tcc_svm_k_fold_k)

        self.calcular_svm_metrics_average()
        self.calcular_roc_auc_score_average()
        self.calcular_fpr_tpr_average()
        self.executar_roc_curve_average()
        self.calcular_confusion_matrix_average()

    def calcular_confusion_matrix_average(self):
        tp_average_soma = 0
        fn_average_soma = 0
        fp_average_soma = 0
        tn_average_soma = 0

        qtde = 0

        for k in self.__k_fold_lista:
            tp_average_soma += k.tp
            fn_average_soma += k.fn
            fp_average_soma += k.fp
            tn_average_soma += k.tn

            qtde += 1

        self.__tp_average = tp_average_soma / qtde
        self.__fn_average = fn_average_soma / qtde
        self.__fp_average = fp_average_soma / qtde
        self.__tn_average = tp_average_soma / qtde

    def calcular_svm_metrics_average(self):
        self.__svm_metrics_average = TccSvmMetrics()

        accuracy_score_soma = 0
        f1_score_soma = 0
        precision_score_soma = 0
        recall_score_soma = 0
        sensitivity_soma = 0
        specificity_soma = 0

        qtde = 0

        for k in self.__k_fold_lista:
            accuracy_score_soma += k.svm_metrics.accuracy_score
            f1_score_soma += k.svm_metrics.f1_score
            precision_score_soma += k.svm_metrics.precision_score
            recall_score_soma += k.svm_metrics.recall_score
            sensitivity_soma += k.svm_metrics.sensitivity
            specificity_soma += k.svm_metrics.specificity

            qtde += 1

        self.__svm_metrics_average.accuracy_score = accuracy_score_soma / qtde
        self.__svm_metrics_average.f1_score = f1_score_soma /qtde
        self.__svm_metrics_average.precision_score = precision_score_soma / qtde
        self.__svm_metrics_average.recall_score = recall_score_soma / qtde
        self.__svm_metrics_average.sensitivity = sensitivity_soma / qtde
        self.__svm_metrics_average.specificity = specificity_soma /qtde

    def calcular_roc_auc_score_average(self):
        roc_auc_score_average_soma = 0
        qtde = 0

        for k in self.__k_fold_lista:
            roc_auc_score_average_soma += k.roc_curve.roc_auc_score_average

            qtde += 1

        self.__roc_auc_score_average = roc_auc_score_average_soma / qtde

    def calcular_fpr_tpr_average(self,):
        qtde = 0

        self.__fpr_average = []
        self.__tpr_average = []

        for k in self.__k_fold_lista:
            for indice in range(len(k.roc_curve.fpr)):

                if indice + 1 > len(self.__fpr_average):
                    self.__fpr_average.append(k.roc_curve.fpr[indice])
                    self.__tpr_average.append(k.roc_curve.tpr[indice])
                else:
                    self.__fpr_average[indice] += k.roc_curve.fpr[indice]
                    self.__tpr_average[indice] += k.roc_curve.tpr[indice]

            qtde += 1

        for indice in range(len(self.__fpr_average)):
            self.__fpr_average[indice] /= qtde
            self.__tpr_average[indice] /= qtde

    def executar_roc_curve_average(self):
        self.__roc_curve_average = TccRocCurve(self.__classe,fpr=self.__fpr_average, tpr=self.__tpr_average)
        self.__roc_curve_average.ordenar_fpr_tpr()

