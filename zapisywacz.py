#pozwala tworzyc datasety danych tensorflowowych
#ktore sa kompatybilne z klasa model_dnn.py

#zapisuje dane w formacie TFRecords file. Jest on 
#wspomniany w https://www.tensorflow.org/api_guides/python/reading_data 
#pod 'standard tensorflow format', podobno estymator jest zoptymalizowany
#do czytania takiego czegos. 
#, oraz lepiej opisany w https://www.tensorflow.org/tutorials/load_data/tf-records 
# oraz jeszcze lepiej opisane tu
#  https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
# 
# Bardzo wysokopoziomowo format ten pozwala na zapisanie slownikow gdzie
# klucze - stringi to nazwy featurow ( w tym formacie label jest
# traktowany dokladnie tak samo jak feture ), zas wartosci to 
# listy floatow, listy intow lub czystych bitow ( jesli ktos chce zapisac obrazki to chyba
# uzyteczne).
# 
# w tym przypadku klucze zapisanego slownika to 
# - 'l' czyli label, 0 oznacza tlo, zas 1 oznacza sygnal
# - 'n' czyli numeryczne wartosci. Czyli to jest lista
#       wszystkich floatow charakteryzujacych przypadek
#       Pakuje wszystkie te floatowe feature do jednego
#       worka gdyz wowczas powstale pliki binarne sa
#       mniejsze i szybciej sie je czyta i przetwarza 
#       przy trenowaniu.
# - 'c' czyli kategoryczne wartosci. To jest lista intow, gdzie
#        kazdy int to jakas kategoryczna cecha np ladunek czegos. 
#       znowu wrzucone do jednego worka z powodu checi 
#       oszczedzania miejsca i przyspieszenia dzialania trenowania
#
# W folderze z datasetem znajduja sie takze zapisane jsonem listy:
# - W pliku 'n' jest 
#   Lista z zapisanymi nazwami numerycznych featurow. Elementy listy to
#   krotki (technicznie tez listy) (nazwa_feature, index_pocz, index_konc) gdzie
#   index_pocz to index od ktorego zaczynaja sie flaoty tego feature w liscie
#   skonkatenowanych numerycznch feature (czyli pod kluczem 'n' w zapisanym
#   TFRecord pliku)
# - w pliku 'c' jest analogiczny opis kategorycznych danych, to znaczy
#    jest zapisana jsonem lista gdzie elementy to 
#   krotki (nazwa_feature, lista_mozliwych_wartosci_tego_feature)
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

#odczytuje nazwy numerycznych feature dla danych z folderu 'nazwa_folderu'
def wczytaj_typy_numerycznych(nazwa_folderu):
    return wczytaj_json(nazwa_folderu + '/n')

#odczytuje nazwy kategorycznych feature dla danych z folderu 'nazwa_folderu'
def wczytaj_typy_kategorycznych(nazwa_folderu):
    return wczytaj_json(nazwa_folderu + '/c')

#Tworzy pusty folder-dataset. Argumenty to 
#nazwa tworzonego folderu, slownik typu nazwa_feature -> rozmiar_tego_feature
#  (nalezy podac numeryczne feature oraz ich rozmiar, np
#  czteroped pewnie ma rozmiar 4, a masa czegos pewnie 1)  ,
#  c to typu nazwa_feature -> lista_wartosci (
#   czyli jakie beda kategoryczne feature wraz z ich wartosciami).
# oprocz tego, ze tworzy folder to daje 3 rzeczy, ktore sa potrzebne
# w funkcji 'zapisz_jeden_przyklad' to znaczy
# zwraca typy numeryczych feature (czyli wnetrze
# pliku 'n'), typy kategorycznych feature (wnetrze
# pliku 'c'), oraz
# writer ktorym mozna pisac do pliku (nadpisywac chyba)
# gdy skonczymy korzystac z writera warto go zamknac
# komenda writer.close()
def tworz_folder_do_zapisu(nazwa_folderu, n, c):
    assert not os.path.isdir(nazwa_folderu)
    os.mkdir(nazwa_folderu)

    #tworzy liste numerycznych typow
    lista_numerycznych_typow = []
    rozmiar_dotychczasowego = 0
    for k in n.keys():
        lista_numerycznych_typow.append((k, rozmiar_dotychczasowego, rozmiar_dotychczasowego + n[k]))
        rozmiar_dotychczasowego += n[k]
    zapisz_json(lista_numerycznych_typow, nazwa_folderu + "/n")
    
    #tworzy liste kategorycznych typow
    lista_kategorycznch_typow = []
    for k in c.keys():
        lista_kategorycznch_typow.append((k, c[k]))
    zapisz_json(lista_kategorycznch_typow, nazwa_folderu + "/c")
    writer = tf.python_io.TFRecordWriter(nazwa_folderu + '/dane')

    return lista_numerycznych_typow, lista_kategorycznch_typow, writer

#  n to slownik nazwa_numerycznego_feature -> jego_wartosc.
#  jego_wartosc to tablica numpy o shapie (-1,) (czyli 1d) lub float lub
#  lista floatow, c to nazwa_feature_kategorycznego -> wartosc_jego
#  (typu int), l to 0 lub 1
#  lnt, lkt, writer to trojka rzeczy zwracanych przez funkcje
#  tworz_folder_do_zapisu
#  pamietaj by po zapisaniu danych
#  zrobic writer.close() (to znaczy po zapisaniu ostatniego)
def zapisz_jeden_przyklad(n, c, l, lnt, lkt, writer):
    #tworze slownik danych, ktory zostanie zapisany
    # w binarnym pliku TFRecord
    slownik_danych = {}
    #zapisuje label
    slownik_danych['l'] = _int64_feature([l])
    #funkcja robiaca numpy tablice 1d z swojego argumentu
    def poprawiacz_numerycznego(liczba_lub_lista_lub_numpy):
        return list(np.array(liczba_lub_lista_lub_numpy).reshape((-1,)))
    
    #konkatenuje numeryczne feature oraz wrzuca wynik do slonika danych
    lista_numerycznych = []
    for i in range(len(lnt)):
        rozwazany = lnt[i]
        lista_numerycznych += poprawiacz_numerycznego( n[ rozwazany[0] ] )
    slownik_danych['n']= _float_feature(lista_numerycznych)

    #konkatenuje kategoryczne dane i wrzuca do slownika danych
    lista_kategorycznch= []
    for i in range(len(lkt)):
        rozwazany = lkt[i]
        lista_kategorycznch += [c[rozwazany[0]]]
    slownik_danych['c']= _int64_feature(lista_kategorycznch)
    
    #serializuj slownik_danych i zapisz go 
    tr_ex = tf.train.Example(features= tf.train.Features(
        feature = slownik_danych))
    writer.write(tr_ex.SerializeToString())

#tworzy slownik typo wpotrzebnych do odczytania przykladu
def tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu):
    lista_numerycznych = wczytaj_typy_numerycznych(nazwa_folderu)
    
    #laczna liczbe numerycznych feature
    laczna_liczba = lista_numerycznych[len(lista_numerycznych) - 1][2]
    
    lista_kategorycznych = wczytaj_typy_kategorycznych(nazwa_folderu)

    read_features= {}
    read_features['n'] = tf.FixedLenFeature([laczna_liczba], dtype=tf.float32)
    read_features['c'] = tf.FixedLenFeature([len(lista_kategorycznych)], dtype=tf.int64)
    read_features['l'] = tf.FixedLenFeature([1], dtype=tf.int64)
    return read_features

#odczytuje dataset ale bez parsowania do sensownych rzeczy
def odczytaj_dataset(nazwa_folderu):
    dataset = tf.data.TFRecordDataset(nazwa_folderu + '/dane')
    return dataset

#zwraca odczytany dataset ( to znaczy odczytuje
# te zapisane slowniki i tworzy dataset z nich
# ). kazdy przyklad odczytany odzielnie
# wiec jest to dosc wolna metoda
def parsuj_po_jednym(nazwa_folderu):
    read_features = tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu)
    def parsuj(serialized_example):
        return tf.parse_single_example(serialized=serialized_example,
                                        features=read_features)
    dataset = odczytaj_dataset(nazwa_folderu)
    dataset = dataset.map(parsuj)
    return dataset

#podobna do parsuj_po_jednym, ale szybsza. Odczytuje
# od razu wiele przykladow i zwraca batch. argument ile_threadow
# oznacza na ilu threadach ma zachodzic ten proces parsowania
def parsuj_i_batchuj(nazwa_folderu, batch_size, ile_threadow):
    read_features = tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu)
    def parsuj(serialized_examples):
         return tf.parse_example(serialized=serialized_examples,
                                        features=read_features)
    dataset = odczytaj_dataset(nazwa_folderu)
    dataset= dataset.batch(batch_size)
    dataset = dataset.map(parsuj, num_parallel_calls = ile_threadow)
    return dataset


    
    



