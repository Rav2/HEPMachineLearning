# HEPMachineLearning

Project devoted to use of Machine Learning to analyze date from the CMS experiment at LHC.

## Authors:
* Pawel Czajka
* Mateusz Fila
* Rafal Maselek

###Pawla notatki o tym co znacza dane pliki
Teraz pojawila sie Przyklad_nowej_funkcjonalnosci.ipynb ktory zawiera
to co nazwa wskazuje.

Dnn_uniwersalny.py to wazny plik, zawiera klase
bedaca siecia neuronowa ktora potrafi czytac dane binarne zapisane przez
inne klasy projektu oraz zna weak supervised algorithm.

instalacja_roota_i_tensorflowa to uwagi o tym jak to mozna zainstalowac na ubuntu. 
jak teraz zainstalowalem tak na ubuntu 18.04 to dziala wszystkoo, pyroot jak 
i root z konsolki.

Io_tf_binary_general.py to wazny plik. Korzysta z niego Dnn_uniwersalny.py jak 
rowniez trzeba z tej klasy korzystac bezposrednio by tworzyc pliki binarne.
potrafi czytac dane w formie takiej jak wychodzi z read_tree.py co pozwala
latwo zapisac dane ktore odzyskal Rafal. Potrafi czytac takze dane w postaci
zblizonej do postaci jaka mozna odzyskac z danych binarnych zapisanych nia sama
( to jest przez Io_tf_binary_genera.py), co powinno ulatwic "tasowanie"
danych w celu wytworzenia w sztuczny sposob sytuacji z weak supervised learning

I0_tf_binary_general.ipynb zawiera przyklad jak tytulowa klasa tego notebooka
potrafi sie dogadac z read_tree.py


read_tree.py to jest klasa Rafala ktora czyta z roota. 

jest tez Przyklad_weak_supervised_learningu.ipynb ktory to zawiera przyklad
uzycia Io_tf_binary_general pisania na inny sposob do datasetow (write_old bodajze)
oraz przyklad jak Dnn_uniwersalny potrafi robic weak supervised learning.


Jeszcze pewnie sa jakies skrypty cpp ale ja tam nie wiem


