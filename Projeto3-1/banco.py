import cv2
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

vocab = pickle.load( open( "vocab.p", "rb" ) )
descritores = pickle.load( open( "descritores.p", "rb" ) )
hist_banco_i = pickle.load( open( "hist_banco.p", "rb" ) )

def computa_descritores(img):
    img_read = cv2.imread(img)
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img_read, None)
    return des


def contagem (img, vocab):
    des = computa_descritores(img)
    centro = vocab.predict(des)
    posicao = [1 for i in range(vocab.n_clusters)]
    for e in centro:
        posicao[e]+=1
    return posicao

def representa_histograma(img, vocab):
    posicao = contagem(img, vocab)
    plt.bar(range(vocab.n_clusters), posicao,  color="blue")
    plt.show()

def encontra_imagens(imgbusca, descritores = descritores, vocab = vocab, hist_banco_i = hist_banco_i, modo=modo):
    hist_busca = contagem(imgbusca, vocab)
    print("Imagem buscada")
    imgb = cv2.imread(imgbusca)
    new_imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2RGB)
    plt.imshow(new_imgb)
    plt.show()
     
    
    chi_list = []
    for i in range(len(descritores)):
        hist_banco= hist_banco_i[i] 

        n=vocab.n_clusters
        chi_square = 0
        for e in range(n):
            chi_square += ((hist_busca[e] - hist_banco[e])**2)/hist_banco[e]

        chi_list.append([chi_square, i])

    ordenada = sorted(chi_list)

    print("Imagens semelhantes")
    for e in range(5):
        index = ordenada[e][1]
        img = cv2.imread(descritores[index][0:][0])
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(new_img)
        plt.show()



# representa_histograma("101_ObjectCategories/Faces/image_0020.jpg", vocab)

img = input("Nome da imagem + caminho: ")
modo = input("1:Chi-square, 2:OpenCV: ")
encontra_imagens(img)