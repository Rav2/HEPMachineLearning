#Model ktory mozna trenowac na danych przygotowanych przez klase
# zapisywacz. Korzysta z klasy tf.estimator.DNNClassifier . 
# czyli glebokiej sieci neuronowej. 
# Staram sie korzystac z tutorialu 
# https://www.tensorflow.org/guide/performance/datasets
# to znaczy stosuje prefetch ze strony
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
# ktore przygotowuje nowe dane gdy model dalej obrabia poprzednie dane
# jest mozliwosc przygotowania danych na kilku thredach (argument
# ile_threadow w funkcji train)
# oraz daje mozliwosc zrobienia cache dla datasetu. 
# to znaczy argument czy_cache w train. Jesli cachujemy to 
# dane sa zapisane w pierwszej epoce trenowania na ramie
# i pozniej szybciej do nich dotrzec w kolejnych epokach.
# Jesli dataset miesci sie na ramie to warto dac True.

import tensorflow as tf
import numpy as np
import json
import os
import zapisywacz

#ogolne funkcje
def zapisz_json(co,gdzie):
    f=open(gdzie,'w')
    f.write(json.dumps(co))

def wczytaj_json(skad):
    f=open(skad,'r')
    return json.loads(f.read())

# Zmienia nazwe kategorycznych feature tak, by
# nie pomieszaly sie z wykorzystywanymi juz kluczami
# w slowniku
def powieksz_nazwe_kategorycznego(nazwa):
    return 'kat_fet_' + nazwa

#tworzy feature columns dla naszego modelu,
# zalaczam dokumentacje, jesli ktos chce cos zmienic
# https://www.tensorflow.org/guide/feature_columns
def rob_feature_columns(nazwa_zrodla):
    kategoryczne_typy = zapisywacz.wczytaj_typy_kategorycznych(nazwa_zrodla)
    numeryczne_typy = zapisywacz.wczytaj_typy_numerycznych(nazwa_zrodla)
    dlugosc_numerycznych = numeryczne_typy[len(numeryczne_typy) - 1][2]
    wyn = []
    wyn.append(tf.feature_column.numeric_column(
        key = 'n', shape=(dlugosc_numerycznych,)  ))
    for kat in kategoryczne_typy:
        nowy = tf.feature_column.categorical_column_with_vocabulary_list(
            key = powieksz_nazwe_kategorycznego( kat[0]),
            vocabulary_list = kat[1],
            dtype = tf.int32
        )
        #tutaj zamiast indicator column, ktora robi one-hot encoding
        #mozna zrobic 
        #https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column
        #ktora uczy sie embeddingu kategorycznych labelkow w jakims
        #R^n. To jest dobrym pomyslem, gdy jest sporo mozliwych
        # wartosci tego kategorycznego feature. 
        hot = tf.feature_column.indicator_column(nowy)
        wyn.append(hot)
    return wyn

#Dosc uniwersalna funkcja przygotowujaca dane dla modelu. 
# zrodlo_danych to folder przygotowany przez zapisywacz.py
# folder_modelu to folder utworzony funkcja tworz_folder (folder modelu)
# batch_size to batch size
# czy_shuffle to czy tasowac dane przed wrzuceniem do modelu
# ile_threadow to na ilu threadach przygotowywac dane
# czy_repeat to czy brac dane z poczatku gdy skoncza sie z konca
# buffer_size liczy sie gdy czy_shuffle = True, 
#   jak duzy ma byc bufer do tasowania
# czy_cache to czy wczytac dane na ram w celu szybszego ich czyania
def input_fn(zrodlo_danych, folder_modelu, batch_size,  
         czy_shuffle, ile_threadow, czy_repeat, buffer_size,
         czy_cache ):
    def wynikowa():
        mean, var = odczytaj_mean_var(folder_modelu)
        dataset = zapisywacz.parsuj_i_batchuj(zrodlo_danych, batch_size, ile_threadow)
        kategoryczne = zapisywacz.wczytaj_typy_kategorycznych(zrodlo_danych)
        if(czy_repeat):
            dataset = dataset.repeat()
        def normalizacja(slownik):
            slownik['n'] = (slownik['n'] - mean) / (var**0.5)
            return slownik
        def odczep_kategoryczne(slownik):
            kat_tab = slownik.pop('c')
            for i in range(len(kategoryczne)):
                slownik[ powieksz_nazwe_kategorycznego(kategoryczne[i][0])] = kat_tab[:, i]
            return slownik
        def wyrwij_label(slownik):
            l= slownik.pop('l')
            return slownik, l
        dataset = dataset.map(odczep_kategoryczne, num_parallel_calls = ile_threadow)
        dataset = dataset.map(normalizacja,num_parallel_calls = ile_threadow)
        dataset = dataset.map(wyrwij_label,num_parallel_calls = ile_threadow)
        if czy_shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.prefetch(1)
        if czy_cache:
            dataset = dataset.cache()
        return dataset
    return wynikowa

#nazwa_folderu to jak sie bedzie nazywac folder modelu
#glowne_zrodlo_danych to plik z ktorego beda pochodzic dane.
# (uzywam do policzenia statystyk by pozniej moc liczyc normalizacje)
#hidden units to jaka ma byc architektura modelu (np [30, 5] to znaczy, ze
# jest jedna ukryta warstwa co ma 30 neuronow, po niej taka co ma 5, 
# potem jest ta wynikowa warstwa)
# dropout to liczba od 0. do 1., im wieksza tym wieksa
# czesc neuronow jest ignorowana przy uczeniu. zwieksza to 
# regularyzacje modelu. Jesli None to nie ma dropoutu.
# zwraca utworzony model. Mozna ten zwracany model zignorowac.
def tworz_folder(nazwa_folderu, glowne_zrodlo_danych, hidden_units, 
    dropout, na_ilu_statystyki = 10000):
    assert not os.path.isdir(nazwa_folderu)
    os.mkdir(nazwa_folderu)
    zapisz_json(glowne_zrodlo_danych, nazwa_folderu + '/glowne_zrodlo_danych')
    zapisz_json(hidden_units, nazwa_folderu + '/hidden_units')
    zapisz_json(dropout, nazwa_folderu + '/dropout')
    def licz_statystyki(nazwa_zrodla, na_ilu):
        dataset = zapisywacz.parsuj_i_batchuj(nazwa_zrodla, na_ilu, 1)
        iterator = dataset.make_one_shot_iterator()
        tab = iterator.get_next()['n']
        mean, variance = tf.nn.moments(tab, axes=[0])
        return mean, variance
    
    mean, variance = licz_statystyki(glowne_zrodlo_danych, na_ilu_statystyki)
    with tf.Session() as sess:
        mean, variance = sess.run((mean, variance))
        def czy_zera_w_variance(variance):
            for i in range(len(variance)):
                if (variance[i] == 0):
                    return True
            return False
        if (czy_zera_w_variance(variance)):
            assert False
        np.savetxt(nazwa_folderu + '/mean', mean)
        np.savetxt(nazwa_folderu + '/var',variance)
    feature_columns = rob_feature_columns(glowne_zrodlo_danych)
    model = tf.estimator.DNNClassifier(
        hidden_units = hidden_units,
        feature_columns = feature_columns,
        model_dir = nazwa_folderu + '/model',
        n_classes = 2,
        dropout = dropout
    )
    #trzeba cos potrenowac by powstaly losowe poczatkowe wagi 
    # modelu
    model.train(
        input_fn = input_fn(glowne_zrodlo_danych, nazwa_folderu,1,False, 1,False, 1, False),
        steps = 1
    )
    return model
    
#pozwala odczytac zapisane statyki danych
def odczytaj_mean_var(nazwa_folderu):
    mean = np.loadtxt(nazwa_folderu + '/mean')
    var = np.loadtxt(nazwa_folderu + '/var')
    return mean, var

#odczytujemy juz istniejacy model 
def wczytaj_model(nazwa_folderu):
    assert os.path.isdir(nazwa_folderu + '/model')
    glowne_zrodlo_danych = wczytaj_json(nazwa_folderu + '/glowne_zrodlo_danych')
    feature_columns = rob_feature_columns(glowne_zrodlo_danych)
    hidden_units = wczytaj_json(nazwa_folderu + '/hidden_units')
    dropout = wczytaj_json(nazwa_folderu + '/dropout')
    return tf.estimator.DNNClassifier(
        hidden_units = hidden_units,
        feature_columns = feature_columns,
        model_dir = nazwa_folderu + '/model',
        n_classes = 2,
        dropout = dropout
    )

#trenuje model. 
# zrodo_danych to folder zapisany przy pomocy zapisywacz.py
# folder_modelu to folder_modelu 
# steps to ile batches trzeba wczytac
# czy_cache to czy wczytac dane do ram. jesli dataset
#        miesci si ena ramie to warto to zrobic.
# 
# reszta argumentow opisana przy okazji input_fn. 
def train(zrodlo_danych, folder_modelu, steps, czy_cache, batch_size = 128,
         czy_shuffle = True, ile_threadow = 1, czy_repeat = True, buffer_size = 1000
         ):
    model = wczytaj_model(folder_modelu)
    model.train(
        input_fn = input_fn(zrodlo_danych, folder_modelu, batch_size,  
         czy_shuffle, ile_threadow, czy_repeat, buffer_size,
         czy_cache ),
        steps = steps
    )

# ewaluuje model
# zrodo_danych to folder zapisany przy pomocy zapisywacz.py
# folder_modelu to folder_modelu
# reszta opisana przy okazji input_fn 
def evaluate(zrodlo_danych, folder_modelu, steps, batch_size = 128,  
        ile_threadow = 1):
    model = wczytaj_model(folder_modelu)
    return model.evaluate(
        input_fn = input_fn(zrodlo_danych, folder_modelu, batch_size,  
         czy_shuffle = False, ile_threadow = ile_threadow, czy_repeat = False, buffer_size = 100,
         czy_cache = False)
    )



    