import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import read_tree as rt

import sys
import json
import pickle
"""
ten nowy bedzie uproszczony
__init__(nazwa_folderu,tryb)
    inputs:
            nazwa_folderu: np "pierwszy_dataset" tam bedzie pisac/stamtad bedzie szczytywac. 
                wydaje mi sie, ze musi byc to nazwa bez spacji oraz moze byc na przyklad "folder/subfolder"
                To zapisuje nie tylko same dane ale także pomocnicze dane do wczytywania ich.
            tryb: np 'r' lub 'w' i oznacza czy czytac ('r') czy pisac ('w')

            
write_general(features,l,przykladowy=False,co_ile_flush_file=10) 
        UWAGA to jest dostepne tylko w przypadku trybu 'w'
        
        features: to slwonik features dla jednego przykladu np {"momentum":[1.,0.,5.,7.]}
            moze liczbe lub liste lub np array typu jakiegos int lub jakiegos float
        l: to jest label jego 0 lub 1. Nie jestem pewien czy da się wpisywać po prostu
            tensorflowowe tensory tutaj. na pewno można numpy obiekty. Prawdopodobnie także
            listy pythonowe. 

        wygodna metoda do tego, azeby "potasowac" przyklady z roznych plikow w jeden dataset, bo pisze
        sie przyklady w takiej formie w jakiej sie odczytywalo z tych datasetow. 
        
        przykladowy: typu bool. Używamy niedomyślnej wartości True w przypadku, gdy utworzyliśmy obiekt
            o trybie 'w' write i chcemy powiedzieć mu, jak wyglądać będa dane którymi zamierzamy go karmić. 
            czyli takiemu obiektowi wystarczy podać jeden taki przykładowy. Daje mu to wiedzę o tym, jak 
            dorabiać nowe featchers. To znaczy trzeba podać jeden przykładowy jeśli chcemy potem uzywać
            .engineer_feature w tym trybie 'w'.
        co_ile_flush_file to jest parametr oznaczający jak często mamy wyrzucać do pliku dane. 
        można poeksperymentować z jego wartością. nie wiem ile to ma wynosić. Może duża wartość 
        pozwoli szybciej zapisywać?

write_from_tree(legs,jets,global_params,properties,l,co_ile_flush_file=10,przykladowy=False)
        metoda w prosty sposob korzystajaca z funkcji metody write_general
        dostepna tylko jak tryb to jest 'w'
            legs,jets,global_params,properties: jak w wyjsciu klasy read_tree, tylko "dla jednego przypadku"
                wiec jest tak
                        zakladam, ze legs to jest lista o shapie (?,4) wypelniona floatami
                        jets dokladnie tak samo
                        global_params to jest w postaci {nazwa:liczba, ...}
                        properties tak samo
                        l to label jest intem rownym to 0 lub 1, gdzie 1 oznacza, ze to jest raczej bardziej ciekawy przypadek
                         a 0 to taki bardziej tlo. to jest int 
                 co_ile_flush_file: to znaczy jak czesto ma oprozniac swoj buffer, liczy sie tylko
                    gdy tryb=='w', nie wiem ile ma wynosic,wiec jak wiesz to smialo ustaw
        
                parametr przykladowy sluzy do tego, ze jak chcemy uzyc .engineer_feature() w trybie 'w' to 
        potrzeba podac przykladowe dane zeby obiekt wiedzial, czy dorabianie featcherow
        jest poprawne. To bedzie ulatwiac uzytkownikowi zycie. 
    
                    
write_from_tree_general(self,legs_list,properties_list,l,co_ile_flush_file=10,przykladowy=False)
        nieco bardziej ogolna wersja poprzedniej metody, gdzie 
        legs oraz jets jako obiekty tych samych typow zostaly wrzócone do tej samej
        listy, tak samo global_params oraz properties. czyli zeby 
        wywolac to tak jak poprzednia metode to piszemy
        obiekt.write_from_tree_general([legs,jets],[global_params,properties],l)
        i to robi to co poprzednia metoda, ale jes to epsion bardziej ogolne. 
        znow jest zbudowane na bazie metody .write_general, wiec latwo zmienic. 
        
                parametr przykladowy sluzy do tego, ze jak chcemy uzyc .engineer_feature() w trybie 'w' to 
        potrzeba podac przykladowe dane zeby obiekt wiedzial, czy dorabianie featcherow
        jest poprawne. To bedzie ulatwiac uzytkownikowi zycie. 

    

close()
            dostepna tylko dla tryb=='w'
            zamyka bezpiecznie nasz plik, tak by mozna go bylo uzyc przy czytaniu
read()
            dostepna tylko dla tryb=='r'
            wyrzuci z siebie tensorflowowy dataset gotowy do uczenia
                w przyszlosci Teraz dorabiam to, ze bedzie z siebire wyrzucac wraz z 
                nowymi dorobionymi featcherami
            dataset wyglada tak, ze pojedynczy przypadek to jest (slownik_featurow,label)
                gdzie label to bedzie 0 lub 1
                zas slownik featerow to jest tak, ze sa klucze stringow a wartosci to jest
                1d arraye ktore sa intami lub floatami 
                to znaczy, ze zwraca to samo co ta stara moja klasa czytajaca
types()
            uzywamy takiego czegos tylko dla type=='r'
            zwraca slownik typow naszego datasetu. to znaczy, ze zwraca slownik
            {'nazwa_featchera':(4,'f')} jesli featcher o tej nazwie to lista 4 floatow
            




engineer_feature(self,f,slownik,typ,nazwa,naprawde_zapisz=True):
        To mozna uzywac w trybie 'w' jak i w trybie 'r', ale znaczenie jest inne. 
        w trybie 'r' dziala tak, ze bedzie nam dorabiac featchers 
        tak 'on the fly' gdy uzyjemy
        metody .read(). 
        Jezeli uzyjemy tej funkcji w trybie 'w' to naszym celem jest 
        dorobienie featcher w zapisanym binarnym pliku i zapisac nowe featchery. 
        Tutaj instrukcja jest taka, ze po inicjalizacji obiektu typu 'w' 
        uzywamy jednej z metod write costam, ale z parametrem 'przykladowy=True'. 
        Teraz mozemy dorabiac featchers przy pomocy tej metody. w trakcie tego 
        mozna sobie patrzeć przy pomocy metody .types() jak się nam zmieniają
        typy featcherów jakie mamy. Nastepnie po zakończeniu tych czynnosći
        zaczynamy uzywać normalnie metody .write costam bez parametru przykladowy. 
        
        Chciałem zrobić coś takiego, że w trybie 'w' podaje się do wyboru czy 
        zapisywać powstały featcher na stałe w pliku z danymi czy jakoś zapamiętać że takie
        coś ma być dorabiane w locie przy wczytywaniu, ale chyba nie da się 
        
        
       
        
        z powodzeniem uzywac wczesniej zrobionych features do produkcji jeszcze nowszych.
        nazwa: czyli jak ten nowy ma sie nazywac
        
        f: to funkcja przyjmująca argumenty o nazwach ze zbioru kluczy slownika slownik, 
                zwraca zas nowy feature (czyli tensor o ksztalcie (-1,).
                w trybie 'r'
                musi to byc funkcja
                dzialajaca dobrze na tensorach z tensorflow. To jest tak fajnie zaklepane, ze
                wyrzuci blad jesli funkcja jest niepoprawna od razu przy wywolaniu tej 
                engineer_feature.
                 funkcja f musi byc taka, ze dobrze dziala na tensorflowywch
                 tensorach o ksztalcie (-1,) to znaczy 
                scisle jednowymiarowych. Wynikiem tej funkcji czyli nowym featurem 
                musi byc znowu tensor o ksztalcie
                (-1,). 
                
                w trybie 'w' ta funkcja 'f' ma dzialac nie na tensorflowowe tensory
                a na takiego typu tensory, co sa w naszych danych ktorymi karmimy
                w .write_general. 
                
                w trybie 'r' jak podacie zla funkcje f to metoda engineer_feature od 
                razu wyrzuca blad, zas w trybie 'w' nie wyrzuca od razu bledu dopiero
                przy zapisywani pojawi się jakis dziwny blad. 
        
        
                https://www.tensorflow.org/api_guides/python/math_ops
                tu macie podstawowe operacje. pamietjcie, ze * oraz + tez mozna uzywac, ale
                nie wszystkie funkcje z numpy sa dobre w tensorflow( to znaczy inaczej sie w nim nazywaja).

        
        slownik:  to slownik którego klucze sa ze zbioru nazw argumentow funkcji f zas 
            zas wartosci to sa nazwy rzeczy wystepujacych w kluczach slownika z metody .types()
            i to mowi jakie nalezy rzeczy z dataset wstawic do funkcji f azeby otrzymac nowy feature
        typ: wynosi np (4,'f') cyzli ze ten nowy
            feature bedzie mial 4 floaty. moze byc tez 'i'. oznacza, 
            czy to co powstaje bedzie intem czy floatem. 
            Tak wiem to leniwe, ale bardziej bugoodporne po mojej stronie. 
        naprawde_zapisz: odnosi sie tylko do przypadku, gdy tryb to 'w' i oznacza odpowiedz na pytanie,
            czy zapisac na dysku nasz nowy featcher czy tylko zapisac informacje o tym jak go odwtorzyc przy wczytywaniu. 
            wazne
            najpierw podajecie te ktore maja byc zapisane na stale. potem podajecie te ktora
            maja byc zapisane na niby to znaczy naprawde_zapisz=False. Jest to po to, azeby
            nie bylo tak, ze przy czytaniu datasetu trzeba korzystac z zmiennych ktorych nie ma.
            Dodatkowo jest tak, ze jak dodajemy featcher na niby to w nim ta funkcja f musi 
            czytać tensory typu tensorflow, zas jesli to jest featcher na prawde to 
            wowczas ta funkcja f ma miec tylo taka wlasnosc, ze dziala dobrze na 
            tensory wrzucane do metod write costam. 
            

            
       
            






"""
class Na_niby_featcher(object):
    def __init__(self, eng):
        self.eng=eng
    def to_dict(self):
        return self.eng


class Io_tf_binary_general:
    def __init__(self,nazwa_folderu,tryb):
        
        self.nazwa_folderu=nazwa_folderu
        def odczytaj_na_niby_featcheres():
            with open(self.nazwa_folderu+'/on_the_fly_featcheres.pkl', 'rb') as input:
                wyrzut=[]
                rob=True
                while rob:
                    try:
                        wyrzut.append(pickle.load(input))
                    except:
                        rob=False
                return wyrzut
        self.tryb=tryb
        
        if tryb=='r':
            slownik_typow=Io_tf_binary_general.wczytaj_json(self.nazwa_folderu+"/metadata")
            self.wewnetrzny=Io_tf_binary_general.Io_tf_binary_stary(
                self.nazwa_folderu+"/dane",slownik_typow,self.tryb)
            na_niby_featchers=odczytaj_na_niby_featcheres()
            for eng in na_niby_featchers:
                self.wewnetrzny.engineer_feature(**(eng.to_dict()))
            
        
        self.nowopowstala=True
        if self.tryb=='w':
            self.new_featcheres_naprawde_zapisz=[]
            self.new_featcheres_na_niby_zapisz=[]
            self.pojawilo_sie_na_niby=False
            self.typy_naprawde={}
            self.typy_naniby={}
            self.typy_pierwsze={}
            self.juz_poznane=False
        
    def engineer_feature(self,f,slownik,typ,nazwa,naprawde_zapisz=True):
        if self.tryb=='r':
            assert not (nazwa in self.wewnetrzny.types().keys())
            self.wewnetrzny.engineer_feature(f,slownik,typ,nazwa)
        else:
            assert not (nazwa in self.przyklad.keys())
            podstawienia={}
            for k in slownik.keys():
                podstawienia[k]=self.przyklad[slownik[k]]
            self.przyklad[nazwa]=f(**podstawienia)
            #self.typy[nazwa]=typ
            if naprawde_zapisz:
                assert self.pojawilo_sie_na_niby==False
                self.new_featcheres_naprawde_zapisz.append(
            {'f':f,'slownik':slownik,'typ':typ,'nazwa':nazwa})
                self.typy_naprawde[nazwa]=typ
                self.typy_naniby[nazwa]=typ
            else:
                self.pojawilo_sie_na_niby=True
                self.new_featcheres_na_niby_zapisz.append(
            {'f':f,'slownik':slownik,'typ':typ,'nazwa':nazwa})
                self.typy_naniby[nazwa]=typ
            
        
        
    
        
        
    
    def zapisz_json(co,gdzie):
        f=open(gdzie,'w')
        f.write(json.dumps(co))
    def wczytaj_json(skad):
        f=open(skad,'r')
        return json.loads(f.read())
    
    def zrob_slownik_typow(legs,jets,global_params,properties):
        #zakladam, ze legs to jest lista o shapie (?,4) wypelniona floatami
        #jets dokladnie tak samo
        #global_params to jest w postaci {nazwa:liczba, ...}
        wyrzut={}
        legs=np.array(legs)
        jets=np.array(jets)
        n_legs=legs.shape[0]
        n_jets=jets.shape[0]
        def czy_int(x):
            type(x)==int or type(x)==int ==numpy.int64

        for i in range(n_legs):
            wyrzut["leg_"+str(i)+"_momentum"]=(4,'f')
        for i in range(n_jets):
            wyrzut["jet_"+str(i)+"_momentum"]=(4,'f')
        def dorob(wyrzut,global_params):
            for param_key in global_params.keys():
                if np.issubdtype(type(global_params[param_key]), np.integer):
                    wyrzut[param_key]=(1,'i')
                else:
                    wyrzut[param_key]=(1,'f')
            return wyrzut
        wyrzut=dorob(wyrzut,global_params)
        wyrzut=dorob(wyrzut,properties)
        return wyrzut
    def zrob_slownik_typow_old_format(f):
        wyrzut={}
        ff={}
        for k in f.keys():
            ff[k]=np.array(f[k]).reshape((-1,))
        for k in ff.keys():
            if np.issubdtype(type(ff[k][0]), np.integer):
                wyrzut[k]=(len(ff[k]),'i')
            else:
                wyrzut[k]=(len(ff[k]),'f')
        return wyrzut
        
    
    def zrob_sensowna_forme(legs,jets,global_params,properties,l):
        #l jest intem i to 0 lub 1
        legs=np.array(legs)
        jets=np.array(jets)
        n_legs=legs.shape[0]
        n_jets=jets.shape[0]

        f={}


        for i in range(n_legs):
            f["leg_"+str(i)+"_momentum"]=legs[i,:]
        for i in range(n_jets):
            f["jet_"+str(i)+"_momentum"]=jets[i,:]

        for param_key in global_params.keys():
            f[param_key]=[global_params[param_key]]
        for k in properties:
            f[k]=[properties[k]]
        return f,l
    
    
    def zrob_sensowna_forme_general(legs_list,properties_list,l):
        #l jest intem i to 0 lub 1
        for i in range(len(legs_list)):
            legs_list[i]=np.array(legs_list[i])
#         n_legs=legs.shape[0]
#         n_jets=jets.shape[0]

        f={}

        for j in range(len(legs_list)):
            for i in range(legs_list[j].shape[0]):
                f["leg_"+str(j)+"_"+str(i)+"_momentum"]=legs_list[j][i,:]


        for i in range(len(properties_list)):
            properties=properties_list[i]
            for k in properties:
                f[k+"_"+str(i)]=[properties[k]]
        return f,l
    
    
    
    def write_from_tree(self,legs,jets,global_params,properties,l,co_ile_flush_file=10,przykladowy=False):
        #zakladam, ze legs to jest lista o shapie (?,4) wypelniona floatami
        #jets dokladnie tak samo
        #global_params to jest w postaci {nazwa:liczba, ...}
        #l jest intem i to 0 lub 1
        assert self.tryb=='w'
#         if self.nowopowstala==True:
#             os.system("mkdir "+self.nazwa_folderu)
#             self.nowopowstala=False
#             slownik= Io_tf_binary_general.zrob_slownik_typow(legs,jets,global_params,properties)
#             self.typy_pierwszego=slownik
#             Io_tf_binary_general.zapisz_json(slownik,self.nazwa_folderu+"/metadata")
#             self.stary_io=Io_tf_binary_general.Io_tf_binary_stary(
#                 self.nazwa_folderu+"/dane",slownik,self.tryb,co_ile_flush_file)
        f,l=Io_tf_binary_general.zrob_sensowna_forme(legs,jets,global_params,properties,l)
        #slownik= Io_tf_binary_general.zrob_slownik_typow_old_format(f)
        #assert self.typy_pierwszego==slownik
        
        
        self.write_general(f,l,przykladowy=przykladowy)
    
    
    def write_from_tree_general(self,legs_list,properties_list,l,co_ile_flush_file=10,przykladowy=False):
        #zakladam, ze legs to jest lista o shapie (?,4) wypelniona floatami
        #jets dokladnie tak samo
        #global_params to jest w postaci {nazwa:liczba, ...}
        #l jest intem i to 0 lub 1
        assert self.tryb=='w'
#         if self.nowopowstala==True:
#             os.system("mkdir "+self.nazwa_folderu)
#             self.nowopowstala=False
#             slownik= Io_tf_binary_general.zrob_slownik_typow(legs,jets,global_params,properties)
#             self.typy_pierwszego=slownik
#             Io_tf_binary_general.zapisz_json(slownik,self.nazwa_folderu+"/metadata")
#             self.stary_io=Io_tf_binary_general.Io_tf_binary_stary(
#                 self.nazwa_folderu+"/dane",slownik,self.tryb,co_ile_flush_file)
        f,l=Io_tf_binary_general.zrob_sensowna_forme_general(legs_list,properties_list,l)
        #slownik= Io_tf_binary_general.zrob_slownik_typow_old_format(f)
        #assert self.typy_pierwszego==slownik
        
        
        self.write_general(f,l,przykladowy=przykladowy)
    
    
        
    def write_general(self,features,l,co_ile_flush_file=10,przykladowy=False):
        
        def zrob_plik_na_niby_featchers():

            with open(self.nazwa_folderu+'/on_the_fly_featcheres.pkl', 'wb') as output:
                for eng in self.new_featcheres_na_niby_zapisz:
                    naniby=Na_niby_featcher(eng)
                    pickle.dump(naniby, output, pickle.HIGHEST_PROTOCOL)
            

                
        
        
        assert self.tryb=='w'
        if not przykladowy:
            if self.nowopowstala==True:
                os.system("mkdir "+self.nazwa_folderu)
                self.nowopowstala=False
                slownik= Io_tf_binary_general.zrob_slownik_typow_old_format(features)
                if  (self.typy_pierwsze=={}):
                    self.typy_pierwszego=slownik
                else:
                    self.typy_pierwszego=self.typy_pierwsze
                if not self.juz_poznane:
                    self.typy_naprawde=Io_tf_binary_general.zrob_slownik_typow_old_format(features)
                    self.typy_naniby=self.typy_naprawde.copy()
                    self.typy_pierwsze=self.typy_naprawde.copy()
                Io_tf_binary_general.zapisz_json(self.typy_naprawde,self.nazwa_folderu+"/metadata")
                self.stary_io=Io_tf_binary_general.Io_tf_binary_stary(
                    self.nazwa_folderu+"/dane",slownik,self.tryb,co_ile_flush_file)
                zrob_plik_na_niby_featchers()
            slownik= Io_tf_binary_general.zrob_slownik_typow_old_format(features)
            assert self.typy_pierwszego==slownik


            self.stary_io.wpisz(features,l,self.new_featcheres_naprawde_zapisz)
        else:
            assert self.nowopowstala
            assert self.juz_poznane==False
            self.juz_poznane=True
            self.przyklad=features
            self.typy_naprawde=Io_tf_binary_general.zrob_slownik_typow_old_format(self.przyklad)
            self.typy_naniby=self.typy_naprawde.copy()
            self.typy_pierwsze=self.typy_naprawde.copy()
            
        
            
            
        
        
        
        
    def close(self):
        assert self.tryb=='w'
        if not self.nowopowstala:
            self.stary_io.close()
    def read(self):
        assert self.tryb=='r'
        
        return self.wewnetrzny.wczytaj_dataset()
    def types(self):
        if self.tryb=='r':
            return self.wewnetrzny.types()
        return self.typy_naniby
        
    
    #jakby ktos kopiowal to to idzie dalej
    #==========================================================    
        """
    do creatora dajemy sobie nazwe pliku (sciezke) oraz slownik typu {'czterowektor': (4,'f'),'intowa_wlasnosc': (1,'i'),
    ...}
    to znaczy nazwe, ile to jest liczb, jakiego typu. Obsluguje na razie jedynie 'f' oraz 'i' to 
    jest float oraz int
    nie mozna uzyc jako klucz 'label', bo to jest wykorzystywana nazwa.


    funkcja wpisz bierze jako argument jeden przypadek  cos typu 
    ({"czterowektor":[1.,2.,3.,4.], ...},1) gdzie 1 jest labelem, label jest intem.
    . oczywiście klucze slownika zgadzają się 
    z kluczami ze slownika ktorego uzylismy do kreatora.

    dataset wczytany metoda wczytaj dataset jest juz w postaci wygodnej dla mnie to znaczy 
    slownik feature, label
    """
    class Io_tf_binary_stary:
        def __init__(self,nazwa_pliku,slownik,tryb,co_ile_flush_file=10):
            #tryb to 'w' dla write oraz 'r' dla read
            
            self.co_ile=co_ile_flush_file
            self.plik=nazwa_pliku
            self.typy=slownik
            for k in self.typy.keys():
                assert self.typy[k][1] in ['f','i']
                assert self.typy[k][0]>0 
                assert np.issubdtype(type(self.typy[k][0]), np.integer)
            self.tryb=tryb
            if tryb=='w':
                self.writer= tf.python_io.TFRecordWriter(self.plik)
            if tryb=='r':
                self.dataset=self.wczytaj_bez_feature_engeeneringu()
            self.liczba_wrzuconych=0

            #self.cos=Io_tf_binary.wrap_int64([5])
        def types(self):
            return self.typy

        def close(self):
            self.writer.close()


        """
        To moze sobie czytac ktos kto chce zmieniac wnetrznosci tej klasy
        Nie polecam

        Teraz ta funkcja wpisz jest ważna. ona bierze po jednym przypadku testowym, 
        ( to jest ta petla for i in range()) i go zapisuje. trzeba zwracac 
        uwage na to jakiego typu sa zapisywane rzeczy. mozna oczywiscie zrobic slownik
        data dluzszym, jesli to w jakis sposob ulatwi nam myslenie o naszych danych. 
        Bo te nasze dane to bedzie slownik list, w ktorych to listach rzeczy maja 
        juz taki sam typ, a klucze to beda jakies opisowe nazwy.
        np 

        data={
        'czteroped_lewej_nogi_czy_cos': wrap_float64(cztero), # gdzie cztero to jest tensor floatow o shape (4,)
        # reszta rzeczy

        }

        Jak byscie chcieli jako wartosci miec stringi to musicie pomyslec jak zrobic wrapy dla stringow. oczywiscie
        nie znajdziecie zadnej dokumentacji.

        UWAGA 
        w tym slowniku data musi byc to co klasyfikujemy oznaczone przy pomocy 'label' bo inaczej sie  wywali program.





        """


        def wpisz(self,features,label,new_featcheres):
            """tworzy ten nasz dataset w pliku out_path 
            tu format jest taki jak byl na poczatku to znaczy taki slownik"""
            f=features
            l=label
            
            for fet in new_featcheres:
                podstawienia={}
                for k in fet['slownik'].keys():
                    podstawienia[k]=f[fet['slownik'][k]]
                f[fet['nazwa']]=fet['f'](**podstawienia)
                self.typy[fet['nazwa']]=fet['typ']
            
            
            def wrap_int64(value):
                """lista intow musi wlesc"""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            def wrap_float64(value):
                """lista floatow musi wlesc"""
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))

            #f,l=kolko_w_kolku() #mozna zmienic jak sie podoba
            def data_slownik(f,l):
                wyrzut={}
                for k in self.typy.keys():
                    if self.typy[k][1]=='f':
                        wyrzut[k]=wrap_float64(np.array(f[k]).reshape((-1,)))
                    else:
                        wyrzut[k]=wrap_int64(np.array(f[k]).reshape((-1,)))
                wyrzut['label']=wrap_int64([l])
                return wyrzut



            #feature=f[i]
            #label=l[i]
            #data = {
             #    'feature': wrap_float64(feature),
            #  'label': wrap_int64([label])
               #     }
            data=data_slownik(f,l)
            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            self.writer.write(serialized)
            self.liczba_wrzuconych+=1
            if self.liczba_wrzuconych%self.co_ile==0:
                self.writer.flush()





        def wczytaj_bez_feature_engeeneringu(self):

            def zeslownikoj(x):
                keys=list(x.keys())
                f={}
                for k in keys:
                    if not k=='label':
                        f[k]=x[k]
                return f,x['label']
            def features_generoj():
                wyrzut={}
                for k in self.typy.keys():
                    if self.typy[k][1]=='f':
                        wyrzut[k]=tf.FixedLenFeature([self.typy[k][0]], tf.float32)
                    else:
                        wyrzut[k]=tf.FixedLenFeature([self.typy[k][0]], tf.int64)
                wyrzut['label']=tf.FixedLenFeature([], tf.int64)
                return wyrzut

            def parse(serialized):

                # Define a dict with the data-names and types we expect to
                # find in the TFRecords file.
                # It is a bit awkward that this needs to be specified again,
                # because it could have been written in the header of the
                # TFRecords file instead.
                """
                features = \
                    {
                        'dwuwektor': tf.FixedLenFeature([2], tf.float32),#z jakiegos powodu to jest float32, nie wiem czemu
                        'label': tf.FixedLenFeature([], tf.int64)
                    }
                """
                features=features_generoj()
                print(features)

                # Parse the serialized data so we get a dict with our data.
                parsed_example = tf.parse_single_example(serialized=serialized,
                                                         features=features)


                return zeslownikoj(parsed_example)

            dataset = tf.data.TFRecordDataset(self.plik)
            dataset = dataset.map(parse)
            return dataset


        def engineer_feature(self,f,slownik,typ,nazwa):
            print("taki tam engeenerowany featcher ")
            print(nazwa)
            print(typ)
            print(slownik)
            assert self.tryb=='r'
            """ to ma zmienic po prostu nasz self.dataset"""
            def dodaj_jeden_feature(engineered,dataset):
                """ten engineered to jest ten slownik {'f':f,'slownik':slownik,'typ':typ,'nazwa':nazwa}"""
                for k in engineered['slownik'].keys():
                    assert engineered['slownik'][k] in self.typy.keys()
                assert not (engineered['nazwa'] in self.typy.keys())
                def lambdowata(f,label):
                    #features,label=jeden_przyklad
                    features=f.copy()
                    def zrob_podstawienie():
                        podstawienie={}
                        for zmienna in engineered['slownik'].keys():
                            podstawienie[zmienna]=features[engineered['slownik'][zmienna]]
                        return podstawienie
                    nazwa=engineered['nazwa']
                    features[nazwa]=engineered['f'](**zrob_podstawienie())
                    self.typy[nazwa]=engineered['typ']
                    return features,label
                return dataset.map(lambdowata)
                    
            engi={'f':f,'slownik':slownik,'typ':typ,'nazwa':nazwa}
            self.dataset= dodaj_jeden_feature(engi,self.dataset)
        def wczytaj_dataset(self):
            assert self.tryb=='r'
            return self.dataset
    
    
    
    
    
    
    
    
    
    