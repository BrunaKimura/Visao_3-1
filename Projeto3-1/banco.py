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

def encontra_imagens(imgbusca, descritores = descritores, vocab = vocab, hist_banco_i = hist_banco_i):
    hist_busca = contagem(imgbusca, vocab)
    print("Imagem buscada")
    imgb = cv2.imread(imgbusca)
    new_imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 10))
    
   
    plt.subplot(3,3,2).axis('off')
    plt.title("imagem buscada")
    plt.imshow(new_imgb)

     
    
    chi_list = []
    if modo==1:
        for i in range(len(descritores)):
            hist_banco= hist_banco_i[i] 

            n=vocab.n_clusters
            chi_square = 0
            for e in range(n):
                chi_square += ((hist_busca[e] - hist_banco[e])**2)/hist_banco[e]

            chi_list.append([chi_square, i])
    
    else:
        hist1 = cv2.calcHist([imgb],[0],None,[256],[0,256])

        for i in range(len(descritores)):
            img_banco = cv2.imread(descritores[i][0:][0])
            hist2 = cv2.calcHist([img_banco],[0],None,[256],[0,256])
            chi_square = cv2.compareHist( hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            chi_list.append([chi_square, i])

    ordenada = sorted(chi_list)

    for e in range(5):
        index = ordenada[e][1]
        print("{0}° imagem semelhante: ".format(e+1), descritores[index][0:][0])
        img = cv2.imread(descritores[index][0:][0])
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(3,3,e+4).axis('off')
        plt.title("{0}° imagem semelhante: ".format(e+1))
        plt.imshow(new_img)
    plt.show()



# representa_histograma("101_ObjectCategories/Faces/image_0020.jpg", vocab)
a = True
img = input("Nome da imagem + caminho: ")
modo = int(input("Qual modo deseja fazer o cálculo? (1:Chi-square, 2:OpenCV): "))
if (modo == 1) or (modo == 2):
    a = False

while a:
    if (modo == 1) or (modo == 2):
        a = False
    else:
        print("digite apenas '1' ou '2'")
        modo = int(input("Qual modo deseja fazer o cálculo? (1:Chi-square, 2:OpenCV): "))

encontra_imagens(img)