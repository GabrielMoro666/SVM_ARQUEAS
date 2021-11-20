import os


class TccExport:

    def __init__(self, classe, kernel = ""):
        self.__classe = classe
        self.__kernel = kernel

        self.__diretorio = ""

        self.configurar_diretorio()

    @property
    def diretorio(self):
        return self.__diretorio

    @diretorio.setter
    def diretorio(self, diretorio):
        self.__diretorio = diretorio

    def configurar_diretorio(self):
        diretorio = self.__classe + "/" + self.__kernel

        if not self.existe_diretorio(diretorio):
            self.criar_diretorio(diretorio)

        self.__diretorio = diretorio

    def existe_diretorio(self, diretorio = ""):
        if diretorio == "":
            diretorio = self.__diretorio

        if diretorio == "":
            return

        return os.path.isdir(diretorio)

    def criar_diretorio(self, diretorio = ""):
        if diretorio == "":
            diretorio = self.__diretorio

        if diretorio == "":
            return

        if not self.existe_diretorio():
            os.makedirs(diretorio)

    def remover_diretorio(self, diretorio = ""):
        if diretorio == "":
            diretorio = self.__diretorio

        if diretorio == "":
            return

        if self.existe_diretorio():
            os.removedirs(diretorio)

    def existe_arquivo(self, arquivo_nome):
        return os.path.exists(self.__diretorio + "/" + arquivo_nome)

    def criar_arquivo(self, arquivo_nome):
        if not self.existe_arquivo(arquivo_nome):
            arquivo = open(self.__diretorio + "/" + arquivo_nome, "w")
            arquivo.close()

    def escrever_arquivo(self, arquivo_nome, texto):
        if not self.existe_arquivo(arquivo_nome):
            self.criar_arquivo(arquivo_nome)

        arquivo = open(self.__diretorio + "/" + arquivo_nome, "a")
        arquivo.write(texto+"\n")
        arquivo.close()

    def remover_arquivo(self, arquivo_nome):
        if self.existe_arquivo(arquivo_nome):
            os.remove(self.__diretorio + "/" + arquivo_nome)

