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

def wczytaj_ig_typy_numerycznych(nazwa_folderu):
    return wczytaj_json(nazwa_folderu + '/ig_n')

def wczytaj_ig_typy_kategorycznych(nazwa_folderu):
    return wczytaj_json(nazwa_folderu + '/ig_c')

#Tworzy pusty folder-dataset. Argumenty to 
#nazwa tworzonego folderu, slownik typu nazwa_feature -> rozmiar_tego_feature
#  (nalezy podac numeryczne feature oraz ich rozmiar, np
#  czteroped pewnie ma rozmiar 4, a masa czegos pewnie 1)  ,
#  c to typu nazwa_feature -> lista_wartosci (
#   czyli jakie beda kategoryczne feature wraz z ich wartosciami).
#ig_n oraz ig_c to te same rzeczy, tyle ze to sa ignorowane 
#numeryczne oraz kategoryczne rzeczy (przy uczeniu ma na nie nie patrzec)
#dodatkowo dajemy is_this_classification, ktory odpowiada na pytanie, 
#czy labelki to 0 1 czy moze sa to ciagle rzeczy. 
# oprocz tego, ze tworzy folder to daje 3 rzeczy, ktore sa potrzebne
# w funkcji 'zapisz_jeden_przyklad' to znaczy
# zwraca typy numeryczych feature (czyli wnetrze
# pliku 'n'), typy kategorycznych feature (wnetrze
# pliku 'c'), oraz
# to samo dla ig_n oraz ig_c oraz
# writer ktorym mozna pisac do pliku (nadpisywac chyba)
# gdy skonczymy korzystac z writera warto go zamknac
# komenda writer.close()
def tworz_folder_do_zapisu(nazwa_folderu, n, c, ig_n, ig_c, is_this_classification):
    assert not os.path.isdir(nazwa_folderu)
    os.mkdir(nazwa_folderu)

    zapisz_json(is_this_classification, nazwa_folderu + "/is_this_classification") 

    #tworzy liste numerycznych typow
    lista_numerycznych_typow = []
    ig_lista_numerycznych_typow = []
    rozmiar_dotychczasowego = 0
    for k in n.keys():
        lista_numerycznych_typow.append((k, rozmiar_dotychczasowego, rozmiar_dotychczasowego + n[k]))
        rozmiar_dotychczasowego += n[k]
    for k in ig_n.keys():
        ig_lista_numerycznych_typow.append((k, rozmiar_dotychczasowego, rozmiar_dotychczasowego + ig_n[k]))
        rozmiar_dotychczasowego += ig_n[k]
    zapisz_json(lista_numerycznych_typow, nazwa_folderu + "/n")
    zapisz_json(ig_lista_numerycznych_typow, nazwa_folderu + "/ig_n")
    
    #tworzy liste kategorycznych typow
    lista_kategorycznch_typow = []
    ig_lista_kategorycznch_typow = []
    for k in c.keys():
        lista_kategorycznch_typow.append((k, c[k]))
    for k in ig_c.keys():
        ig_lista_kategorycznch_typow.append((k, ig_c[k]))
    zapisz_json(lista_kategorycznch_typow, nazwa_folderu + "/c")
    zapisz_json(ig_lista_kategorycznch_typow, nazwa_folderu + "/ig_c")
    writer = tf.python_io.TFRecordWriter(nazwa_folderu + '/dane')

    return lista_numerycznych_typow, lista_kategorycznch_typow, ig_lista_numerycznych_typow, \
        ig_lista_kategorycznch_typow, writer

#  n to slownik nazwa_numerycznego_feature -> jego_wartosc.
#  jego_wartosc to tablica numpy o shapie (-1,) (czyli 1d) lub float lub
#  lista floatow, c to nazwa_feature_kategorycznego -> wartosc_jego
#  (typu int), l to 0 lub 1 jesli is_this_classification, lub
#  float w przeciwnym przypadku, 
#  lnt, lkt, writer to trojka rzeczy zwracanych przez funkcje
#  tworz_folder_do_zapisu
#  pamietaj by po zapisaniu danych
#  zrobic writer.close() (to znaczy po zapisaniu ostatniego)
def zapisz_jeden_przyklad(n, c, ig_n, ig_c, l, lnt, lkt, ig_lnt, ig_lkt, writer, is_this_classification):
    #tworze slownik danych, ktory zostanie zapisany
    # w binarnym pliku TFRecord
    slownik_danych = {}
    #zapisuje label
    if (is_this_classification):
        slownik_danych['l'] = _int64_feature([l])
    else:
        slownik_danych['l'] = _float_feature([l])
    #funkcja robiaca numpy tablice 1d z swojego argumentu
    def poprawiacz_numerycznego(liczba_lub_lista_lub_numpy):
        return list(np.array(liczba_lub_lista_lub_numpy).reshape((-1,)))
    
    #konkatenuje numeryczne feature oraz wrzuca wynik do slonika danych
    lista_numerycznych = []
    for i in range(len(lnt)):
        rozwazany = lnt[i]
        lista_numerycznych += poprawiacz_numerycznego( n[ rozwazany[0] ] )
    slownik_danych['n']= _float_feature(lista_numerycznych)
    ig_lista_numerycznych = []
    for i in range(len(ig_lnt)):
        rozwazany = ig_lnt[i]
        ig_lista_numerycznych += poprawiacz_numerycznego( ig_n[ rozwazany[0] ] )
    slownik_danych['ig_n']= _float_feature(ig_lista_numerycznych)

    #konkatenuje kategoryczne dane i wrzuca do slownika danych
    lista_kategorycznch= []
    for i in range(len(lkt)):
        rozwazany = lkt[i]
        lista_kategorycznch += [c[rozwazany[0]]]
    slownik_danych['c']= _int64_feature(lista_kategorycznch)
    ig_lista_kategorycznch = []
    for i in range(len(ig_lkt)):
        rozwazany = ig_lkt[i]
        ig_lista_kategorycznch += [ig_c[rozwazany[0]]]
    slownik_danych['ig_c']= _int64_feature(ig_lista_kategorycznch)
    
    #serializuj slownik_danych i zapisz go 
    tr_ex = tf.train.Example(features= tf.train.Features(
        feature = slownik_danych))
    writer.write(tr_ex.SerializeToString())

#tworzy slownik typo wpotrzebnych do odczytania przykladu
def tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu):
    is_this_classification = wczytaj_json(nazwa_folderu + "/is_this_classification")
    lista_numerycznych = wczytaj_typy_numerycznych(nazwa_folderu)

    ig_lista_numerycznych = wczytaj_ig_typy_numerycznych(nazwa_folderu)

    len_num = ""
    len_ig_num = ""
    
    if (len(lista_numerycznych) == 0):
        len_num = 0
    else:
        len_num = lista_numerycznych[len(lista_numerycznych) - 1][2]
    
    if (len(ig_lista_numerycznych) == 0):
        len_ig_num = 0
    else:
        len_ig_num = ig_lista_numerycznych[len(ig_lista_numerycznych) - 1][2] - len_num 
    
    lista_kategorycznych = wczytaj_typy_kategorycznych(nazwa_folderu)
    ig_lista_kategorycznych = wczytaj_ig_typy_kategorycznych(nazwa_folderu)

    read_features= {}
    read_features['n'] = tf.FixedLenFeature([len_num], dtype=tf.float32)
    read_features['ig_n'] = tf.FixedLenFeature([len_ig_num], dtype=tf.float32)
    read_features['c'] = tf.FixedLenFeature([len(lista_kategorycznych)], dtype=tf.int64)
    read_features['ig_c'] = tf.FixedLenFeature([len(ig_lista_kategorycznych)], dtype=tf.int64)
    if is_this_classification:
        read_features['l'] = tf.FixedLenFeature([1], dtype=tf.int64)
    else:
        read_features['l'] = tf.FixedLenFeature([1], dtype=tf.float32)
    return read_features

#odczytuje dataset ale bez parsowania do sensownych rzeczy
def odczytaj_dataset(nazwa_folderu):
    dataset = tf.data.TFRecordDataset(nazwa_folderu + '/dane')
    return dataset

#zwraca odczytany dataset ( to znaczy odczytuje
# te zapisane slowniki i tworzy dataset z nich
# ). kazdy przyklad odczytany odzielnie
# wiec jest to dosc wolna metoda
#zwraca zarowno ignorowane jak i nie ignorowane
def parsuj_po_jednym_all(nazwa_folderu):
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
#zwraca zarowno ignorowane jak i nie ignorowane. 
def parsuj_i_batchuj_all(nazwa_folderu, batch_size, ile_threadow):
    read_features = tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu)
    def parsuj(serialized_examples):
         return tf.parse_example(serialized=serialized_examples,
                                        features=read_features)
    dataset = odczytaj_dataset(nazwa_folderu)
    dataset= dataset.batch(batch_size)
    dataset = dataset.map(parsuj, num_parallel_calls = ile_threadow)
    return dataset

def drop_ignored(s):
    s.pop('ig_n', None)
    s.pop('ig_c', None)
    return s

#zwraca odczytany dataset ( to znaczy odczytuje
# te zapisane slowniki i tworzy dataset z nich
# ). kazdy przyklad odczytany odzielnie
# wiec jest to dosc wolna metoda
#zwraca tylko nie ignorowane
def parsuj_po_jednym_not_ignored(nazwa_folderu):
    read_features = tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu)
    def parsuj(serialized_example):
        return tf.parse_single_example(serialized=serialized_example,
                                        features=read_features)
    dataset = odczytaj_dataset(nazwa_folderu)
    dataset = dataset.map(parsuj)
    dataset = dataset.map(drop_ignored)
    return dataset

#podobna do parsuj_po_jednym, ale szybsza. Odczytuje
# od razu wiele przykladow i zwraca batch. argument ile_threadow
# oznacza na ilu threadach ma zachodzic ten proces parsowania
#zwraca zarowno tylko nie ignorowane
def parsuj_i_batchuj_not_ignored(nazwa_folderu, batch_size, ile_threadow):
    read_features = tworz_slownik_typow_potrzebny_do_parsowania(nazwa_folderu)
    def parsuj(serialized_examples):
         return tf.parse_example(serialized=serialized_examples,
                                        features=read_features)
    dataset = odczytaj_dataset(nazwa_folderu)
    dataset= dataset.batch(batch_size)
    dataset = dataset.map(parsuj, num_parallel_calls = ile_threadow)
    dataset = dataset.map(drop_ignored)
    return dataset


    
    



