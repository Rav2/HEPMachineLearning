import tensorflow as tf
import numpy as np
import json
import os

#na poczatek ogolne funkcje

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

#nazwa tworzonego folderu, slownik typu nazwa -> rozmiar, c to typu nazwa -> lista_wartosci
# zwraca typy numeryczych danych, typy kategorycznych danych, oraz
#writer ktorym mozna pisac do pliku (nadpisywac chyba)
#te zwracane uzywa sie w zapisz_jeden_przyklad
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

# n to slownik nazwa -> lista floatow lub (byc moze)
#  numpy tablica shapu (k,) tez floatow, c to nazwa -> wartosc (typu int), l to 0 lub 1
#pamietaj by po wszystkim zrobic writer.close() (to znaczy po zapisaniu ostatniego)
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

#tworzy slownik typo wpotrzebnych do parsowania
def tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu):
    lista_numerycznych = wczytaj_json(nazwa_folderu + '/n')
    laczna_liczba = lista_numerycznych[len(lista_numerycznych) - 1][2]
    lista_kategorycznych = wczytaj_json(nazwa_folderu + '/c')

    read_features= {}
    read_features['n'] = tf.FixedLenFeature([laczna_liczba], dtype=tf.float32)
    read_features['c'] = tf.FixedLenFeature([len(lista_kategorycznych)], dtype=tf.int64)
    read_features['l'] = tf.FixedLenFeature([1], dtype=tf.int64)
    return read_features

#odczytuje dataset ale bez parsowania do sensownych rzeczy
def odczytaj_dataset(nazwa_folderu):
    dataset = tf.data.TFRecordDataset(nazwa_folderu + '/dane')
    return dataset

#parsuje kazdy przyklad z osobna. jest wiec wolna metoda
def parsuj_po_jednym(nazwa_folderu):
    read_features = tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu)
    def parsuj(serialized_example):
        return tf.parse_single_example(serialized=serialized_example,
                                        features=read_features)
    dataset = odczytaj_dataset(nazwa_folderu)
    dataset = dataset.map(parsuj)
    return dataset

#od razu batchuje przy parsowaniu, wiec szybsza
#podajemy ile threadow ma robic ten preprocesing by bylo szybciej
def parsuj_i_batchuj(nazwa_folderu, batch_size, ile_threadow):
    read_features = tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu)
    def parsuj(serialized_examples):
         return tf.parse_example(serialized=serialized_examples,
                                        features=read_features)
    dataset = odczytaj_dataset(nazwa_folderu)
    dataset= dataset.batch(batch_size)
    dataset = dataset.map(parsuj, num_parallel_calls = ile_threadow)
    return dataset


    
    



