# HEPMachineLearning

Project devoted to use of Machine Learning to analyze date from the CMS experiment at LHC.

## Authors:
* Pawel Czajka
* Mateusz Fila
* Rafal Maselek

## Doxygen doxumentation:
The C++ part of the project is accompanied with Doxygen documentation that user can generate by its own.
To generate the documentation please follow the instruction:
1) Install Doxygen. On Debian-like operating systems:
`sudo apt-get install doxygen`
2) Install LateX and graphviz if you don't have it:
`sudo apt-get install texlive-full`
`sudo apt-get install graphviz`
3) Now, execute the following command in the main directory of the repository:
`doxygen Doxyfile`
It will build LateX and html output to Docs/ directory. Doxyfile is a file which tells doxygen what to do. Feel free to modify it in order to adjust the documentation to your needs.
4) For interactive html output open Docs/html/index.html via any web browser.
5) In order to generate pdf output, move in terminal to Docs/latex directory:
`cd Docs/latex`
and use the following command:
`make pdf`
A new file called "refman.pdf" will be created in Docs/latex directory.
