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

# The following functions can be used to convert a value to a type compatible
# with tf.Example. values to lista takich rzeczy.

def _bytes_feature(values):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

#nazwa tworzonego, slownik typu nazwa -> rozmiar, c to typu nazwa -> lista_wartosci
# zwraca writer sluzacy do pisania do tego czegos
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
    writer = tf.python_io.TFRecordWriter(nazwa_folderu + '/dane')
    return lista_numerycznych_typow, lista_kategorycznch_typow, writer

# n to slownik nazwa -> lista lub numpy tablica shapu (k,), c to nazwa -> wartosc, l to 0 lub 1
#pamietaj by po wszystkim zrobic writer.close()
def zapisz_jeden_przyklad(n, c, l, lnt, lkt, writer):
    slownik_danych = {}
    slownik_danych['l'] = _int64_feature([l])
    
    lista_numerycznych = []
    for i in range(len(lnt)):
        rozwazany = lnt[i]
        lista_numerycznych += n[rozwazany[0]]
    slownik_danych['n']= _float_feature(lista_numerycznych)

    lista_kategorycznch= []
    for i in range(len(lkt)):
        rozwazany = lkt[i]
        lista_kategorycznch += [c[rozwazany[0]] ]
    slownik_danych['c']= _int64_feature(lista_kategorycznch)
    

    tr_ex = tf.train.Example(features= tf.train.Features(
        feature = slownik_danych))
    writer.write(tr_ex.SerializeToString())


def odczytaj_dataset(nazwa_folderu):
    dataset = tf.data.TFRecordDataset(nazwa_folderu + '/dane')
    return dataset



    
    



