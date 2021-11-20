import datetime

from TccExport import TccExport
from TccRocCurve import TccRocCurve
from TccSvm import TccSvm


def main():

    timestamp = str(datetime.datetime.now()).replace(":", "")

    # hvo
    linear = executar_svm("hvo_and_shuffled.csv", "hvo", "label", k_fold=10, decision_limite=0.5, kernel="linear");
    exportar(linear, timestamp)
    rbf = executar_svm("hvo_and_shuffled.csv", "hvo", "label", k_fold=10, decision_limite=0.5, kernel="rbf")
    exportar(rbf, timestamp)
    poly = executar_svm("hvo_and_shuffled.csv", "hvo", "label", k_fold=10, decision_limite=0.5, kernel="poly")
    exportar(poly, timestamp)

    #merge grafico
    merge = TccRocCurve(classe="HVO")
    merge.add_curva("Linear",linear.fpr_average,linear.tpr_average)
    merge.add_curva("RBF", rbf.fpr_average, rbf.tpr_average)
    merge.add_curva("Polinomial", poly.fpr_average, poly.tpr_average)
    exportar_merge_grafico(merge,timestamp)

    # sso
    linear = executar_svm("sso_and_shuffled.csv", "sso", "label", k_fold=10, decision_limite=0.5, kernel="linear")
    exportar(linear, timestamp)
    rbf = executar_svm("sso_and_shuffled.csv", "sso", "label", k_fold=10, decision_limite=0.5, kernel="rbf")
    exportar(rbf, timestamp)
    poly = executar_svm("sso_and_shuffled.csv", "sso", "label", k_fold=10, decision_limite=0.5, kernel="poly")
    exportar(poly, timestamp)

    # merge grafico
    merge = TccRocCurve(classe="SSO")
    merge.add_curva("Linear", linear.fpr_average, linear.tpr_average)
    merge.add_curva("RBF", rbf.fpr_average, rbf.tpr_average)
    merge.add_curva("Polinomial", poly.fpr_average, poly.tpr_average)
    exportar_merge_grafico(merge, timestamp)

    # tk
    linear = executar_svm("tk_and_shuffled.csv", "tk", "label", k_fold=10, decision_limite=0.5, kernel="linear")
    exportar(linear, timestamp)
    rbf = executar_svm("tk_and_shuffled.csv", "tk", "label", k_fold=10, decision_limite=0.5, kernel="rbf")
    exportar(rbf, timestamp)
    poly = executar_svm("tk_and_shuffled.csv", "tk", "label", k_fold=10, decision_limite=0.5, kernel="poly")
    exportar(poly, timestamp)

    # merge grafico
    merge = TccRocCurve(classe="TK")
    merge.add_curva("Linear", linear.fpr_average, linear.tpr_average)
    merge.add_curva("RBF", rbf.fpr_average, rbf.tpr_average)
    merge.add_curva("Polinomial", poly.fpr_average, poly.tpr_average)
    exportar_merge_grafico(merge, timestamp)


def executar_svm(arquivo, classe, posicao_classe, k_fold=10, decision_limite=0.5, kernel="rbf"):
    tcc_svm = TccSvm(arquivo, classe, posicao_classe)

    tcc_svm.kernel = kernel
    tcc_svm.k_fold = k_fold
    tcc_svm.decision_limite = decision_limite

    tcc_svm.executar_k_fold()

    return tcc_svm

def exportar_merge_grafico(tcc_roc_curve, timestemp):
    export = TccExport(tcc_roc_curve.classe.upper() + "/" + timestemp)

    tcc_roc_curve.plotar_roc_curve(nome_grafico=tcc_roc_curve.classe.upper(), multicurvas=True, exportar=True, diretorio=export.diretorio, nome_arquivo="ROC CURVE MERGE.tiff")

def exportar(tcc_svm, timestemp):
    nome_arquivo = "REPORT.TXT"

    export = TccExport(tcc_svm.classe.upper() + "/" + timestemp, tcc_svm.kernel.upper())

    export.criar_arquivo(nome_arquivo)

    export.escrever_arquivo(nome_arquivo, tcc_svm.classe.upper() + " REPORT")
    export.escrever_arquivo(nome_arquivo,"---------------------------------------------------------------------------")
    export.escrever_arquivo(nome_arquivo, "")

    export.escrever_arquivo(nome_arquivo, "AVERAGE METRICS")
    export.escrever_arquivo(nome_arquivo, "ACCURACY SCORE AVERAGE: %.4f" % (tcc_svm.svm_metrics_average.accuracy_score))
    export.escrever_arquivo(nome_arquivo, "F1 SCORE AVERAGE: %.4f" % (tcc_svm.svm_metrics_average.f1_score))
    export.escrever_arquivo(nome_arquivo, "PRECISION SCORE AVERAGE: %.4f" % (tcc_svm.svm_metrics_average.precision_score))
    export.escrever_arquivo(nome_arquivo, "RECALL SCORE AVERAGE: %.4f" % (tcc_svm.svm_metrics_average.recall_score))
    export.escrever_arquivo(nome_arquivo, "SENSITIVITY AVERAGE: %.4f" % (tcc_svm.svm_metrics_average.sensitivity))
    export.escrever_arquivo(nome_arquivo, "SPECIFICITY AVERAGE: %.4f" % (tcc_svm.svm_metrics_average.specificity))
    export.escrever_arquivo(nome_arquivo, "ROC AUC AVERAGE %.4f" % (tcc_svm.roc_auc_score_average))

    export.escrever_arquivo(nome_arquivo, "")

    export.escrever_arquivo(nome_arquivo, "AVERAGE CONFUSION MATRIX")
    export.escrever_arquivo(nome_arquivo, "TP AVERAGE: %s" % (tcc_svm.tp_average))
    export.escrever_arquivo(nome_arquivo, "FN AVERAGE: %s" % (tcc_svm.fn_average))
    export.escrever_arquivo(nome_arquivo, "FP AVERAGE: %s" % (tcc_svm.fp_average))
    export.escrever_arquivo(nome_arquivo, "TN AVERAGE: %s" % (tcc_svm.tn_average))

    tcc_svm.roc_curve_average.plotar_roc_curve(exportar=True, diretorio=export.diretorio, nome_arquivo="ROC CURVE AVERAGE.tiff")

    export.escrever_arquivo(nome_arquivo, "")

    export.escrever_arquivo(nome_arquivo, "K-FOLDS")
    export.escrever_arquivo(nome_arquivo, "---------------------------------------------------------------------------")

    for k in range(len(tcc_svm.k_fold_lista)):
        export.escrever_arquivo(nome_arquivo, "")
        export.escrever_arquivo(nome_arquivo, "K-FOLD " + str(k+1))

        export.escrever_arquivo(nome_arquivo, "")

        export.escrever_arquivo(nome_arquivo, "METRICS")
        export.escrever_arquivo(nome_arquivo, "ACCURACY SCORE: %.4f" % (tcc_svm.k_fold_lista[k].svm_metrics.accuracy_score))
        export.escrever_arquivo(nome_arquivo, "F1 SCORE: %.4f" % (tcc_svm.k_fold_lista[k].svm_metrics.f1_score))
        export.escrever_arquivo(nome_arquivo, "PRECISION SCORE: %.4f" % (tcc_svm.k_fold_lista[k].svm_metrics.precision_score))
        export.escrever_arquivo(nome_arquivo, "RECALL SCORE: %.4f" % (tcc_svm.k_fold_lista[k].svm_metrics.recall_score))
        export.escrever_arquivo(nome_arquivo, "SENSITIVITY: %.4f" % (tcc_svm.k_fold_lista[k].svm_metrics.sensitivity))
        export.escrever_arquivo(nome_arquivo, "SPECIFICITY: %.4f" % (tcc_svm.k_fold_lista[k].svm_metrics.specificity))
        export.escrever_arquivo(nome_arquivo, "ROC AUC AVERAGE %.4f" % (tcc_svm.k_fold_lista[k].roc_curve.roc_auc_score_average))

        export.escrever_arquivo(nome_arquivo, "")

        export.escrever_arquivo(nome_arquivo, "CONFUSION MATRIX")
        export.escrever_arquivo(nome_arquivo, "TP: %s" % (tcc_svm.k_fold_lista[k].tp))
        export.escrever_arquivo(nome_arquivo, "FN: %s" % (tcc_svm.k_fold_lista[k].fn))
        export.escrever_arquivo(nome_arquivo, "FP: %s" % (tcc_svm.k_fold_lista[k].fp))
        export.escrever_arquivo(nome_arquivo, "TN: %s" % (tcc_svm.k_fold_lista[k].tn))

        tcc_svm.k_fold_lista[k].roc_curve.plotar_roc_curve(exportar=True, diretorio=export.diretorio, nome_arquivo="ROC CURVE K-FOLD " + str(k+1) + ".tiff")


if __name__ == "__main__":
    main()
