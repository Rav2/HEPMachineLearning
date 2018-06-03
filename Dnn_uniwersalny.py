

"""


Tym razem jak podajemy sciezke do folderu to ten plik ma juz byc pelen danych. 

bardzo nie lubi jak mu sie przerywa trening przy pomocy kernel interrrupt
pozniej nie chodza rozne rzeczy w takim przerwanym obiekcie
bo jak zaczyna trening to finalizuje graph co kolwiek to znacyzy, i 
jak konczy to chyba go odfinalizowywuje, ale jak przerwiemy to tego nie zrobi. nie do 
konca to rozumiem. wiec nie przerywamy treningu ani niczego inngo.

Na razie w pierwszym rzucie nie bedzie jeszcze danych typu kategorycznego.

To ma byc kompatybilne z klasa Io_tf_binary_general



__init__(nazwa_folderu,hidden_units,model_dir,czy_nowy=True)
        to jest nazwa_folderu odnosi sie do folderu w ktorym pisala klasa
        Io_tf_binary_general czy cos
        hidden_units to jest lista po ile ma byc ukrytych unitsow. czyli nie podajemy rozmiaru
        danych wejsciowych ani wyjsciowych. idziemy od pierwszej (najblizszej inputu) do ostatniej
        model_dir to tam bedzie pisac swoje rzeczy nasz model
        czy_nowy: to znaczy, czy tworzymy nowy model czy probojemy wczytac model ktory
        kiedys sie liczyl i ma juz dosyc dobre wagi? (sa subtelnosci z uzywaniem
        enginner_featcher na poziomie tego uniwersalnego dnn_estimatora, to znaczy
        trzeba dac takie same enginner featcher jak ma dzialac.)
engineer_feature(self,f,slownik,typ,nazwa)
        dokladnie jak w tamtym dla io_tf_binary_general
make_model(self,cathegorical_vocabulary_lists):
        to tworzy nasz model po tym jak podawalismy transformacje dla danych zgodnie z 
        engineer_feature() (taka jest metoda)
        tutaj cathegorical_vocabulary_lists to jest slownik typu
        {'jakies_id':[1,2,3]} to znaczy, ze takie cos moze miec takie wartosci. w kluczach
        sa tylko kategoryczne argumenty. Wazne, ze to maja byc integery a nie na przyklad
        floaty czy inne takie. stringow tez nie obsluguję. 
        
        Teraz dodaje normalizacje danych wejsciowych. odbedzie sie w trakcie wykonywania
        make_model. Normalizuje tylko dane floatowe, zakladajac, ze te intowe 
        to sa jakies znormalizowane.
train 
        jest self explainatory. wydaje mi sie, ze to robi tak, ze kontynuuje
        trenowanie z miejsca w ktorym skonczylo
evaluate 
        to jest zwykla ewaluacja. tyle, ze mozna podac jako argument "folder" z ktorego pochodza nasze dane.
        ale ta funkcja zawsze dziala tak, ze po prostu traktuje 1 jako prawdziwe przypadki a 
        0 jako tlo
ma tez rysowac roc czy cos takiego. 
niech evaluate daje nam zbior punktow na krzywej roc a potem 
niech bedzie metoda rotate na przyklad ktora nam zrobi krzywa roc dla odrozniania prawdziwych klas
evaluate_jak_z_pracy(self,p0to1,p1to1,ile_take=10000,folder="")
        to robi takie szacowanie krzywej roc oraz auc wartosci jak w tej pracy. tutaj podajemy
        jako folder te nasze przypadki. ile_take to znaczy jak wiele przypadkow z tego datasetu
         z argumentu 'folder' nalezy wziasc. p0to1 to jest prawdopodoienstwo, ze przypadek oznaczony 
         0 jest tak na prawde 1 zas p1to1 to jest ze oznaczony jako 1 jest tak naprawde wlasnie rozpadem 
         czyli 1.zwraca jako pierwsze auc a potem jeszcze 3 listy zwiazane z rysowaniem krzywej roc.
         to znaczy liste jej xsow, jej ygrekow oraz jakim tresholdom te punkty krzywej roc odpowiadaja.
types()
        to jest zmienna tego obiektu ktora ma informacje o tym jakie feateres sa w naszym datasecie
        po tym jak zrobisz feature engineering to sie zmieni wynik podzialania .types()

        
niech w trakcie train niech pojawi sie accuracy co step.
mozna by sprobopwac tensorboarda. dodac do tensorboard zmienne do monitorowania.
Tu jest taki artefakt, ze ten area under curve czasem jest więcej niz 1 i to nie za dobrze.
.
Dodaj normalizacje oraz zapisz po tym jak przeleci przez dane treningowe. 






"""

from sklearn import metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import Io_tf_binary_general as io
import json
import os
import pickle

class Na_niby_featcher(object):
    def __init__(self, eng):
        self.eng=eng
    def to_dict(self):
        return self.eng

class Dnn_uniwersalny:
    def __init__(self,nazwa_folderu,hidden_units,model_dir,czy_nowy=True):
        if czy_nowy:
            assert not os.path.isdir(model_dir)
            
        
        self.not_compiled=True
        if not czy_nowy:
            self.not_compiled=False
        self.model_dir=model_dir
        self.nazwa_folderu=nazwa_folderu
        os.system("mkdir "+model_dir)
        self.hidden_units=hidden_units
        self.wczytywacz=io.Io_tf_binary_general(nazwa_folderu,'r')
        self.lista_transformacji=[]
        self.czy_nowy=czy_nowy
        self.czy_wczytane_typy=False

    def licz_normalizacje(wczytywacz,na_ilu):
        dataset=wczytywacz.read()
        typy=wczytywacz.types()

        zbachowany=dataset.repeat().batch(na_ilu)
        iterator = zbachowany.make_one_shot_iterator()
        para=iterator.get_next()
        def mean_var(para,k):
            return tf.nn.moments(para[0][k],axes=[0])
            # gdzie var to jest to podniesione do kwadratu cos
        mean_slownik={}
        sig_slownik={}
        for k in typy.keys():
            if typy[k][1]=='f':
                mean,var=mean_var(para,k)
                mean_slownik[k]=mean
                sig_slownik[k]=var**(0.5)
        return mean_slownik,sig_slownik

    def zapisz_json(co,gdzie):
        f=open(gdzie,'w')
        f.write(json.dumps(co))
    def wczytaj_json(skad):
        f=open(skad,'r')
        return json.loads(f.read())
    def odczytaj_na_niby_featcheres(self):
        with open(self.model_dir+'/new_featchers.pkl', 'rb') as input:
            wyrzut=[]
            rob=True
            while rob:
                try:
                    wyrzut.append(pickle.load(input).to_dict())
                except:
                    rob=False
            return wyrzut
    def zrob_plik_na_niby_featchers(self):

        with open(self.model_dir+'/new_featchers.pkl', 'wb') as output:
            for eng in self.lista_transformacji:
                naniby=Na_niby_featcher(eng)
                pickle.dump(naniby, output, pickle.HIGHEST_PROTOCOL)
        
    def make_model(self,cathegorical_vocabulary_lists={},na_ilu_liczyc_mean_oraz_sigma=10000):
        if not self.czy_nowy:
            assert cathegorical_vocabulary_lists=={}
        assert self.not_compiled or (not self.czy_nowy)
        if self.czy_nowy:
            self.zrob_plik_na_niby_featchers()
            Dnn_uniwersalny.zapisz_json(cathegorical_vocabulary_lists,self.model_dir+"/cathegorical")
            Dnn_uniwersalny.zapisz_json(self.wczytywacz.types(),self.model_dir+'/typy')
        cathegorical_vocabulary_lists=Dnn_uniwersalny.wczytaj_json(self.model_dir+"/cathegorical")
        self.czy_wczytane_typy=True
        self.typy_wczytane=Dnn_uniwersalny.wczytaj_json(self.model_dir+"/typy")
        
        my_feature_columns = []
        for k in self.typy_wczytane.keys():
            if not( k in cathegorical_vocabulary_lists.keys()):
                if self.typy_wczytane[k][1]=='f':
                    t=tf.float32
                else:
                    t=tf.int32
                my_feature_columns.append(tf.feature_column.numeric_column(key=k,shape=\
                            (self.typy_wczytane[k][0],),dtype=t ))
                
            else:
                assert self.typy_wczytane[k][1]=='i'
                my_feature_columns.append(
                    
                    tf.feature_column.embedding_column(
                    
    tf.feature_column.categorical_column_with_vocabulary_list(
    key=k,vocabulary_list=cathegorical_vocabulary_lists[k],
    ),len(cathegorical_vocabulary_lists[k])))
        self.feature_columns=my_feature_columns
        
        if self.czy_nowy:
            dict_of_means,dict_of_sigmas=Dnn_uniwersalny.licz_normalizacje(self.wczytywacz,
                                na_ilu_liczyc_mean_oraz_sigma)
            with tf.Session() as sess:
                def zlistoj(slownik):
                    wyrzut=slownik.copy()
                    for k in slownik.keys():
                        wyrzut[k]=list(slownik[k])
                        for i in range(len(wyrzut[k])):
                            wyrzut[k][i]=float(wyrzut[k][i])
                    return wyrzut
                Dnn_uniwersalny.zapisz_json(zlistoj(sess.run(dict_of_means)),self.model_dir+"/means")
                Dnn_uniwersalny.zapisz_json(zlistoj(sess.run(dict_of_sigmas)),self.model_dir+"/sigmas")
        
        def znumpyuj(slownik):
            wyrzut=slownik.copy()
            for k in slownik.keys():
                wyrzut[k]=np.array(slownik[k])
            return wyrzut
        dict_of_means=znumpyuj(Dnn_uniwersalny.wczytaj_json(self.model_dir+'/means'))
        dict_of_sigmas=znumpyuj(Dnn_uniwersalny.wczytaj_json(self.model_dir+'/sigmas'))
            
        self.lista_transformacji_wczytana=self.odczytaj_na_niby_featcheres()
            
        def input_fn( batch_size=100,buffer_size=1000,folder=self.nazwa_folderu,one_epoch=False,czy_shuffle=True
                    ,czy_batch=True,take=False,ile_take=10000,kategoryczne=cathegorical_vocabulary_lists.keys(),
                    lista_transformacji=self.lista_transformacji_wczytana,dict_of_means=dict_of_means,
                    dict_of_sigmas=dict_of_sigmas):
            """input function for training
            nazwy to lista nazw feature w kolejnosci wystepowania
            if one_epoch to tylko jedna epoka"""
            io_gen=io.Io_tf_binary_general(folder,'r')
            
            for tran in lista_transformacji:
                io_gen.engineer_feature(**tran)
            dataset=io_gen.read()
                
            def fun(f,l):
                wyrzut=f.copy()
                for k in kategoryczne:
                    wyrzut[k]=tf.reshape(f[k],shape=[])
                return wyrzut,l
            def normalizacja(f,l):
                wyrzut=f.copy()
                for k in dict_of_means.keys():
                    wyrzut[k]=(f[k]-dict_of_means[k])/dict_of_sigmas[k]
                return wyrzut,l
                
            
            dataset=dataset.map(fun)
            dataset=dataset.map(normalizacja)
            if take:
                dataset=dataset.take(ile_take)
            if czy_shuffle:
                dataset=dataset.shuffle(buffer_size)
            if one_epoch:
                dataset=dataset.repeat(1)
            else:
                dataset=dataset.repeat()
            if czy_batch:
                dataset=dataset.batch(batch_size)

            return dataset
        self.input_fn=input_fn
        
        self.classifier = tf.estimator.DNNClassifier(
        feature_columns=self.feature_columns,
        hidden_units=self.hidden_units,
        model_dir=self.model_dir+'/tensorflowowy',
        n_classes=2)
        self.not_compiled=False
    def types(self):
        if self.czy_wczytane_typy:
            return self.typy_wczytane
        return self.wczytywacz.types()
    def engineer_feature(self,f,slownik,typ,nazwa):
        assert self.not_compiled
        self.wczytywacz.engineer_feature(f,slownik,typ,nazwa)
        self.lista_transformacji.append({'f':f,'slownik':slownik,'typ':typ,'nazwa':nazwa})
    def train(self,batch_size=128,buffer_size=1000,steps=3000):
        self.classifier.train(
        input_fn=lambda:self.input_fn( batch_size,buffer_size),
        steps=steps)
    def evaluate(self,batch_size=128,buffer_size=1000,steps=1000,folder=""):
        if folder=="":
            folder=self.nazwa_folderu
        self.last_eval_result = self.classifier.evaluate(
        input_fn=lambda:self.input_fn( batch_size,buffer_size,folder=folder),
        steps=steps)
        return self.last_eval_result
    """
    def input_fn_new_graph( folder,batch_size=100,buffer_size=1000,one_epoch=False,czy_shuffle=True
                    ,czy_batch=True):
        
        g = tf.Graph()
        with g.as_default():
            dataset=io.Io_tf_binary_general(folder,'r').read()
            if czy_shuffle:
                dataset=dataset.shuffle(buffer_size)
            if one_epoch:
                dataset=dataset.repeat(1)
            else:
                dataset=dataset.repeat()
            if czy_batch:
                dataset=dataset.batch(batch_size)
            return dataset
            """
    def _labelki( folder,ile_take):
        """pomocnicza funkcja ktora zwraca generator labelkow przypadkow"""
        #g = tf.Graph()
        #with g.as_default():
        dataset=io.Io_tf_binary_general(folder,'r').read().take(ile_take)

        dataset=dataset.repeat(1)
        dane=dataset
        iterator = dane.make_one_shot_iterator()
        para=iterator.get_next()
        def wyrzut():
            with tf.Session() as sess2:
                while True:
                    try:
                        yield sess2.run(para)[1]
                    except:
                        break
        return wyrzut()
            
            
    
    def evaluate_jak_z_pracy(self,p0to1,p1to1,ile_take=10000,folder=""):
        def generator_of_predictions_and_labels(self,batch_size=128,folder="",ile_take=10000):
            if folder=="":
                folder=self.nazwa_folderu   
            return (self.classifier.predict(input_fn=
               lambda:self.input_fn(batch_size,folder=folder,one_epoch=True,czy_shuffle=False,take=True,ile_take=ile_take)),
                    Dnn_uniwersalny._labelki(folder,ile_take)  )
        #tu jest pewien problem. jak sie odpali ten generator rzeczy wynikajacych z predict to\
        #dopuki sie nie wykona nie da sie nic  robic na grafie. bo jest chwilowo finalized. wiec
        #najpierw odczytam sobie jedne rzeczy a potem drugie. nie moge zrobic zip tych 
        # 2 generatorow. dlatego uzywam funkcji take ktora ograniczy nam to jak duzo danych dostaniemy
        def ta_praw(a):
            wyrzut=[]
            for i in a:
                wyrzut.append(i['probabilities'][1])
            return wyrzut


        def benchmark(gen_z_estimatora,gen_ground_truth_mixed_class,p0to1,p1to1):
            """p0to1 to prawdopodobienstow, ze jak cos ma labelke 0 to to jest 1"""
            #przez wyniki rozumiem liczbe ktora im jest wieksza tym bardziej estymator mysli ze to jest 1
            probabilities_of_being_1=ta_praw(gen_z_estimatora)
            true_class_identity=list(gen_ground_truth_mixed_class)
            #przez class rozumiem tu ta mieszanine z prawdopodpbienstwem

            assert len(probabilities_of_being_1)==len(true_class_identity)

            fpr, tpr, thresholds = metrics.roc_curve(true_class_identity
                        , probabilities_of_being_1, pos_label=1)
            eps_s_list=[]
            eps_b_list=[]
            for i in range(len(fpr)):
                odw=np.linalg.inv(np.array([[p0to1,1-p0to1],[p1to1,1-p1to1]]))
                epss_epsb=np.matmul(odw,np.array([[fpr[i]],[tpr[i]]]))
                eps_s,eps_b=epss_epsb
                eps_s_list.append(eps_s)
                eps_b_list.append(eps_b)
            return eps_s_list,eps_b_list,thresholds
        a,b=generator_of_predictions_and_labels(self,batch_size=128,folder=folder,ile_take=10000)
        eps_s,eps_b,thresholds=benchmark(a,b,p0to1,p1to1)
        auc=metrics.auc(eps_b,eps_s,True)
        print("auc wynosi "+str(auc))
        plt.clf()
        plt.scatter(eps_b,eps_s)
        plt.xlabel("eps_b")
        plt.ylabel("eps_s")
        return auc,eps_s,eps_b,thresholds
    
    
    
    
    
    
        
        
        
    
        