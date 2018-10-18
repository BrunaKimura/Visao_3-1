import cv2
import os, sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

def computa_descritores(img):
    img_read = cv2.imread(img)
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img_read, None)
    return des

def le_descritores_imagens(pastas, max_items=10):
    path = "./101_ObjectCategories/"
    caminho = []
    
    for file in pastas:
        dirs = sorted(os.listdir(path + file))
        
            
        for i in range(max_items):
            img = "101_ObjectCategories/" + file + '/' + dirs[i]
            matriz_discritores = computa_descritores(img)

            caminho.append([img, matriz_discritores])

    return caminho

def contagem (img, vocab):
    des = computa_descritores(img)
    centro = vocab.predict(des)
    posicao = [1 for i in range(vocab.n_clusters)]
    for e in centro:
        posicao[e]+=1
    return posicao


def cria_vocabulario(descritores, sz=300):
    kmeans = KMeans(n_clusters=sz, random_state=0).fit(descritores)
    return kmeans

path = ["Faces", "garfield", "platypus", "nautilus", "elephant", "gerenuk", "flamingo", "crab", "ewer", "laptop", "pizza"]

descritores = le_descritores_imagens(path)

matriz_descri = descritores[0][1:][0]

for i in range(1, len(descritores)):
    matriz_descri = np.vstack((matriz_descri, descritores[i][1:][0]))

vocab = cria_vocabulario(matriz_descri)

hist_banco = []
for i in range(len(descritores)):
        hist_banco.append(contagem(descritores[i][0:][0], vocab))

pickle.dump( vocab, open( "vocab.p", "wb" ) )
pickle.dump( descritores, open("descritores.p","wb"))
pickle.dump( hist_banco, open("hist_banco.p","wb"))
