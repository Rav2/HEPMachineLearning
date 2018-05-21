
"""


Tym razem jak podajemy sciezke do folderu to ten plik ma juz byc pelen danych. 

bardzo nie lubi jak mu sie przerywa trening przy pomocy kernel interrrupt
pozniej nie chodza rozne rzeczy w takim przerwanym obiekcie
bo jak zaczyna trening to finalizuje graph co kolwiek to znacyzy, i 
jak konczy to chyba go odfinalizowywuje, ale jak przerwiemy to tego nie zrobi. nie do 
konca to rozumiem. wiec nie przerywamy treningu ani niczego inngo.

Na razie w pierwszym rzucie nie bedzie jeszcze danych typu kategorycznego.

To ma byc kompatybilne z klasa Io_tf_binary_general

__init__(nazwa_folderu,hidden_units,model_dir)
        to jest nazwa_folderu odnosi sie do folderu w ktorym pisala klasa
        Io_tf_binary_general czy cos
        hidden_units to jest lista po ile ma byc ukrytych unitsow. czyli nie podajemy rozmiaru
        danych wejsciowych ani wyjsciowych. idziemy od pierwszej (najblizszej inputu) do ostatniej
        model_dir to tam bedzie pisac swoje rzeczy nasz model
engineer_feature(self,f,slownik,typ,nazwa)
        dokladnie jak w tamtym dla io_tf_binary_general
make_model(self,cathegorical_vocabulary_lists):
        to tworzy nasz model po tym jak podawalismy transformacje dla danych zgodnie z 
        engineer_feature() (taka jest metoda)
        tutaj cathegorical_vocabulary_lists to jest slownik typu
        {'jakies_id':[1,2,3]} to znaczy, ze takie cos moze miec takie wartosci. w kluczach
        sa tylko kategoryczne argumenty. Wazne, ze to maja byc integery a nie na przyklad
        floaty czy inne takie. stringow tez nie obsluguję. 
train 
        jest self explainatory. wydaje mi sie, ze to robi tak, ze kontynuuje
        trenowanie z miejsca w ktorym skonczylo
evaluate 
        to jest zwykla ewaluacja. tyle, ze mozna podac jako argument "folder" z ktorego pochodza nasze dane.
        ale ta funkcja zawsze dziala tak, ze po prostu traktuje 1 jako prawdziwe przypadki a 
        0 jako tlo
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

        




"""
from sklearn import metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import Io_tf_binary_general as io


class Dnn_uniwersalny:
    def __init__(self,nazwa_folderu,hidden_units,model_dir):
        self.not_compiled=True
        self.model_dir=model_dir
        self.nazwa_folderu=nazwa_folderu
        self.hidden_units=hidden_units
        self.wczytywacz=io.Io_tf_binary_general(nazwa_folderu,'r')
        self.lista_transformacji=[]
        
    def make_model(self,cathegorical_vocabulary_lists={}):
        assert self.not_compiled
        my_feature_columns = []
        for k in self.wczytywacz.types().keys():
            if not( k in cathegorical_vocabulary_lists.keys()):
                if self.wczytywacz.types()[k][1]=='f':
                    t=tf.float32
                else:
                    t=tf.int32
                my_feature_columns.append(tf.feature_column.numeric_column(key=k,shape=\
                            (self.wczytywacz.types()[k][0],),dtype=t ))
            else:
                assert self.wczytywacz.types()[k][1]=='i'
                my_feature_columns.append(
                    
                    tf.feature_column.embedding_column(
                    
    tf.feature_column.categorical_column_with_vocabulary_list(
    key=k,vocabulary_list=cathegorical_vocabulary_lists[k],
    ),len(cathegorical_vocabulary_lists[k])))
        self.feature_columns=my_feature_columns
        
        def input_fn( batch_size=100,buffer_size=1000,folder=self.nazwa_folderu,one_epoch=False,czy_shuffle=True
                    ,czy_batch=True,take=False,ile_take=10000,kategoryczne=cathegorical_vocabulary_lists.keys(),
                    lista_transformacji=self.lista_transformacji):
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
                
            
            dataset=dataset.map(fun)
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
        model_dir=self.model_dir,
        n_classes=2)
        self.not_compiled=False
    def types():
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
    
    
    
    
    
    
        
        
        
    
        