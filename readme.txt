1. instalujemy roota i tensorflowa tak jak w pliku instalacja_roota_i_tensorflowa
  (chyba ze juz mamy zainstalowane dobre wersje)
2. do folderu data_rootowe_obrobione wrzucamy dane.root takie jak mi Mateuszu
 przeslales
3. notebook wczytywanie_na_czysto.ipynb tworzy binarne tensorflowowe
 pliki zawierajace 
 tlo oraz sygnal (te foldery nazywac sie beda
 wlasnie tlo oraz sygnal) ( trzeba w odpowiednim miejscu notebooka
 (3 komorka) wpisac sciezki/nazwy
 tych plikow, btw czy dobrze zdecydowalem co jest sygnalem ( rozpadem higgsa)
 a co tłem?)
 notebook ten robi takze podstawowe obrobki danych, to znaczy liczy 
 logarytm niektorych z nich by byly bardziej plaskie ich rozklady,
 wyrzuca "singularne" dane, jak jakies dane wydawaly sie zdublowane
 ( to znaczy typu [1,1,5,5,2,2] ) to je oddublowalem. Podejrzewam, ze w przypadku
 odrozniania tej innej pary bedzie podobnie
4. notebook zrob_plik_z_mieszanina_tla_oraz_synalu.ipynb dzieli 
 dane na zbior train ( na ktorym sie trenuje estymator), val 
 (na ktorym sie sprawdza czy model sie uogolnia na danych na ktorych 
 nie trenowal) oraz test ( na ktorym ostatecznie podaje sie skutecznosc
 modelu. 
 Przyjalem metodologie, ze skoro przypadkow tla jest znacznie mniej
 niz sygnalu, to wrzucalem wielokrotnie przypadki tla tak, by
 zawsze polowa plikow stanowil sygnal a polowa tlo (sa powkladane na
 zmiane do plikow binarnych)
5. notebook pierwsze_uczenie.ipynb zawiera przyklad, ze da sie
 trenowac. 


cele:
- poprobowac rozne architektury sieci ( czyli w lini 

estymator=dnn.Dnn_uniwersalny("train",[10],"pierwszy_estymator")

pliku pierwsze_uczenie.ipynb mozna [10], (jest to lista
liczebnosci warstw neuronow ukrytych, im wiecej tym 
bardziej skomplikowany model, ktory moze do wszystkiego
sie dopasowac, ale moze miec klopoty z uogolnianiem. 
prawdopodobnie nie warto wiecej niz 3 warstwy z powodu
pewnych problemow glebokich sieci neuronowych)
Krotko mowiac mozna to [10] zmienic na cos innego np [100,10]
Po takiej zmianie nalezy sprawdzic, jak dziala na val

- mozna sie zastanowic nad tymi cloudami. Pewnie na naszychy kompach by
 sie puszczalo cos w stylu wczytywanie_na_czysto.ipynb, a potem na chmurze
 reszte dostarczajac chmurze foldery tlo oraz sygnal

