import tensorflow as tf
import numpy as np
import json
import os
import zapisywacz

def zapisz_json(co,gdzie):
    f=open(gdzie,'w')
    f.write(json.dumps(co))

def wczytaj_json(skad):
    f=open(skad,'r')
    return json.loads(f.read())

def wyrwij_label(slownik):
    l= slownik.pop('l')
    return slownik, l

def rob_feature_columns(nazwa_zrodla):
    kategoryczne_typy = zapisywacz.wczytaj_typy_kategorycznych(nazwa_zrodla)
    numeryczne_typy = zapisywacz.wczytaj_typy_numerycznych(nazwa_zrodla)
    dlugosc_numerycznych = numeryczne_typy[len(numeryczne_typy) - 1][2]
    wyn = []
    wyn.append(tf.feature_column.numeric_column(
        key = 'n', shape=(dlugosc_numerycznych,)  ))
    for kat in kategoryczne_typy:
        nowy = tf.feature_column.categorical_column_with_vocabulary_list(
            key = kat[0],
            vocabulary_list = kat[1],
            dtype = tf.int32
        )
        hot = tf.feature_column.indicator_column(nowy)
        wyn.append(hot)
    return wyn

#nazwa_folderu to jak sie bedzie nazywac folder modelu
#glowne_zrodlo_danych to plik z ktorego beda pochodzic dane
# do policzenia danych do normalizacji danych
def tworz_folder(nazwa_folderu, glowne_zrodlo_danych, hidden_units, na_ilu_statystyki = 10000):
    assert not os.path.isdir(nazwa_folderu)
    os.mkdir(nazwa_folderu)
    def licz_statystyki(nazwa_zrodla, na_ilu):
        dataset = zapisywacz.parsuj_i_batchuj(nazwa_zrodla, na_ilu, 1)
        iterator = dataset.make_one_shot_iterator()
        tab = iterator.get_next()['n']
        mean, variance = tf.nn.moments(tab, axes=[0])
        return mean, variance
    
    mean, variance = licz_statystyki(glowne_zrodlo_danych, na_ilu_statystyki)
    with tf.Session() as sess:
        mean, variance = sess.run((mean, variance))
        np.savetxt(nazwa_folderu + '/mean', mean)
        np.savetxt(nazwa_folderu + '/var',variance)
    feature_columns = rob_feature_columns(glowne_zrodlo_danych)
    model = tf.estimator.DNNClassifier(
        hidden_units = hidden_units,
        feature_columns = feature_columns,
        model_dir = nazwa_folderu + '/model',
        n_classes = 2
    )
    return model
    

def odczytaj_mean_var(nazwa_folderu):
    mean = np.loadtxt(nazwa_folderu + '/mean')
    var = np.loadtxt(nazwa_folderu + '/var')
    return mean, var

#odczytujemy juz istniejacy model
def wczytaj_model(nazwa_folderu):
    assert os.path.isdir(nazwa_folderu + '/model')
    return tf.estimator.DNNClassifier(model_dir = nazwa_folderu + '/model')

def input_fn(zrodlo_danych, folder_modelu, batch_size,  
         czy_shuffle, ile_threadow, czy_repeat, buffer_size,
         czy_cache ):
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
            slownik[kategoryczne[i][0]] = kat_tab[:, i]
        return slownik
    dataset = dataset.map(odczep_kategoryczne, num_parallel_calls = ile_threadow)
    dataset = dataset.map(normalizacja,num_parallel_calls = ile_threadow)
    dataset = dataset.map(wyrwij_label,num_parallel_calls = ile_threadow)
    if czy_shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.prefetch(1)
    if czy_cache:
        dataset = dataset.cache()
    return dataset