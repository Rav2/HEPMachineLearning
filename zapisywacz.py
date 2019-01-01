import tensorflow as tf
import numpy as np
import json
import os

def zapisz_json(co,gdzie):
    f=open(gdzie,'w')
    f.write(json.dumps(co))

def wczytaj_json(skad):
    f=open(skad,'r')
    return json.loads(f.read())

#n to slownik numerycznych, c to slownik nie numerycznych, l to 0 lub 1
def tworz_folder_na_podstawie_przykladowego(nazwa_folderu, n, c):
    assert not os.path.isdir(nazwa_folderu)
    os.mkdir(nazwa_folderu)

    lista_numerycznych_typow = []
    rozmiar_dotychczasowego = 0
    for k in n.keys():
        lista_numerycznych_typow.append((k, rozmiar_dotychczasowego, rozmiar_dotychczasowego + n[k]))
        rozmiar_dotychczasowego += n[k]
    zapisz_json(lista_numerycznych_typow, nazwa_folderu + "/n")
    
    lista_kategorycznch_typow = []
    for k in c.keys():
        lista_kategorycznch_typow.append((k, c[k]))
    zapisz_json(lista_kategorycznch_typow, nazwa_folderu + "/c")





    
    



