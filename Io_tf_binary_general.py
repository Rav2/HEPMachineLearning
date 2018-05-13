"""
ten nowy bedzie uproszczony
__init__(nazwa_folderu,tryb,co_ile_flush_file=10)
    inputs:
            nazwa_folderu np "pierwszy_dataset" tam bedzie pisac/stamtad bedzie sczytywac. 
            tryb np 'r' lub 'w' i oznacza czy czytac chcesz czy pisac
            co_ile_flush_file to znaczy jak czesto ma oprozniac swoj buffer, liczy sie tylko
                    gdy tryb=='w', nie wiem ile ma wynosic,wiec jak wiesz to smialo ustaw

write(legs,jets,global_params,properties,l)
        dostepna tylko jak tryb to jest 'w'
            legs,jets,global_params,properties jak w wyjsciu klasy read_tree, tylko "dla jednego przypadku"
                wiec jest tak
                        zakladam, ze legs to jest lista o shapie (?,4) wypelniona floatami
                        jets dokladnie tak samo
                        global_params to jest w postaci {nazwa:liczba, ...}
                        properties tak samo
                        l to label jest intem rownym to 0 lub 1, gdzie 1 oznacza, ze to jest raczej bardziej ciekawy przypadek
                         a 0 to taki bardziej tlo. to jest int 

close()
            dostepna tylko dla tryb=='w'
            zamyka bezpiecznie nasz plik
read()
            dostepna tylko dla tryb=='r'
            wyrzuci z siebie tensorflowowy dataset gotowy do uczenia
                w przyszlosci read moze przeprowadzac dodawanie nowych featcherkow,ale na razie
                tego nie robi. W sumie jak mamy dataset to mozemy tez te featcherki nowe dorobic poxniej.
                zobaczymy jak to bedzie.
            dataset wyglada tak, ze pojedynczy przypadek to jest (slownik_featurow,label)
                gdzie label to bedzie 0 lub 1
                zas slownik featerow to jest tak, ze sa klucze stringow a wartosci to jest
                1d arraye ktore sa intami lub floatami 
                to znaczy, ze zwraca to samo co ta stara moja klasa czytajaca
types()
            uzywamy takiego czegos tylko dla type=='r'
            zwraca slownik typow naszego datasetu
            
write_old(features,l) 
        features to slwonik features dla jednego przykladu
        moze liczbe lub liste lub np array typu jakiegos int lub jakiegos float
        l to jest label jego 0 lub 1
        to jest dostepne tylko w przypadku trybu 'w'
        
            







"""


class Io_tf_binary_general:
    def __init__(self,nazwa_folderu,tryb,co_ile_flush_file=10):
        
        self.nazwa_folderu=nazwa_folderu
        self.tryb=tryb
        self.co_ile=co_ile_flush_file
        self.nowopowstala=True
        
    
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
    
    def write(self,legs,jets,global_params,properties,l):
        #zakladam, ze legs to jest lista o shapie (?,4) wypelniona floatami
        #jets dokladnie tak samo
        #global_params to jest w postaci {nazwa:liczba, ...}
        #l jest intem i to 0 lub 1
        assert self.tryb=='w'
        if self.nowopowstala==True:
            os.system("mkdir "+self.nazwa_folderu)
            self.nowopowstala=False
            slownik= Io_tf_binary_general.zrob_slownik_typow(legs,jets,global_params,properties)
            self.typy_pierwszego=slownik
            Io_tf_binary_general.zapisz_json(slownik,self.nazwa_folderu+"/metadata")
            self.stary_io=Io_tf_binary_general.Io_tf_binary_stary(
                self.nazwa_folderu+"/dane",slownik,self.tryb,self.co_ile)
        f,l=Io_tf_binary_general.zrob_sensowna_forme(legs,jets,global_params,properties,l)
        slownik= Io_tf_binary_general.zrob_slownik_typow_old_format(f)
        assert self.typy_pierwszego==slownik
        
        
        self.stary_io.wpisz(f,l)
        
    def write_old(self,features,l):
        assert self.tryb=='w'
        if self.nowopowstala==True:
            os.system("mkdir "+self.nazwa_folderu)
            self.nowopowstala=False
            slownik= Io_tf_binary_general.zrob_slownik_typow_old_format(features)
            self.typy_pierwszego=slownik
            Io_tf_binary_general.zapisz_json(slownik,self.nazwa_folderu+"/metadata")
            self.stary_io=Io_tf_binary_general.Io_tf_binary_stary(
                self.nazwa_folderu+"/dane",slownik,self.tryb,self.co_ile)
        slownik= Io_tf_binary_general.zrob_slownik_typow_old_format(features)
        assert self.typy_pierwszego==slownik
        
        
        self.stary_io.wpisz(features,l)
            
            
        
        
        
        
    def close(self):
        assert self.tryb=='w'
        if not self.nowopowstala:
            self.stary_io.close()
    def read(self):
        assert self.tryb=='r'
        slownik=Io_tf_binary_general.wczytaj_json(self.nazwa_folderu+"/metadata")
        stary_io=Io_tf_binary_general.Io_tf_binary_stary(
                self.nazwa_folderu+"/dane",slownik,self.tryb,self.co_ile)
        return stary_io.wczytaj_dataset()
    def types(self):
        if self.tryb=='r':
            return Io_tf_binary_general.wczytaj_json(self.nazwa_folderu+"/metadata")
        if self.nowopowstala==False:
            return self.self.typy_pierwszego
        
    
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
            if tryb=='w':
                self.writer= tf.python_io.TFRecordWriter(self.plik)
            self.liczba_wrzuconych=0

            #self.cos=Io_tf_binary.wrap_int64([5])

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


        def wpisz(self,features,label):
            """tworzy ten nasz dataset w pliku out_path 
            tu format jest taki jak byl na poczatku to znaczy taki slownik"""
            f=features
            l=label
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










        def wczytaj_dataset(self):

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
    
    
    
    
    
    
    
    
    
    
    
