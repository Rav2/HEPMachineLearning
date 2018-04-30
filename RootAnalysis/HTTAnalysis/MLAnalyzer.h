//! Class for extracting data for further ML processing in python.
/*!
  \author Rafal Maselek
  \date May 2018
  
  This class extracts four-momenta of legs and jets from the analysis, together with some global parameters
  and particles' properties. Data is written to a TTree in "Summary" branch of the main analysis file. It
  can be then exported to python and TensorFlow.
*/
#ifndef RootAnalysis_MLAnalyzer_H
#define RootAnalysis_MLAnalyzer_H

#include "Analyzer.h"
#include "AnalysisEnums.h"
#include "MLObjectMessenger.h"
#include "TLorentzVector.h"
#include "HTTEvent.h"
#include <set>

class MLAnalyzer : public Analyzer
{

 public:

  MLAnalyzer(const std::string & aName, const std::string & aDecayMode);

  virtual ~MLAnalyzer();
  
  ///Initialize the analyzer
  virtual void initialize(TDirectory* aDir,
			  pat::strbitset *aSelections){

    //default is to do whatever the analyzer does
    filterEvent_ = false;
};
  

  virtual Analyzer * clone() const{ return 0;}
  
  virtual bool analyze(const EventProxyBase& iEvent);

  virtual bool analyze(const EventProxyBase& iEvent, ObjectMessenger *aMessenger);

  virtual void finalize(){;};

  virtual void clear(){;};

  virtual void addBranch(TTree *tree);

  virtual void addCutHistos(TList *aList){;};

  const std::string & name() const{return myName_;};

  bool filter() const{ return filterEvent_;};

  std::string cfgFileName;

 protected:

  pat::strbitset *mySelections_;

  ///Types of the selection flow
  std::vector<std::string> selectionFlavours_;

 private:

  std::string myName_;
  int execution_counter_; // counts calls of analyze()
  unsigned legs_no_;
  unsigned jets_no_;
  TTree *MLTree_; // stores the address of output TTree


  //should this Analyzer be able to filter events
  bool filterEvent_;

  //containers for four-momenta
  std::vector<double> legs_p4_;
  std::vector<double> jets_p4_;
  // storage for params loaded from file and assignement to concrete legs and jets
  std::map<std::string, std::vector<double>> params_; //current values of params
  std::map<std::string, std::set<unsigned>> params_legs_; //which legs have which params
  std::map<std::string, std::set<unsigned>> params_jets_; //which jets have which params
  // global parameters of the analylis
  int nJets30_;
  float visMass_;
  float higgsPT_;
  float betaScore_;
  float higgsMassTrans_;

  void extractP4(const TLorentzVector& v, const std::string destination, const unsigned no);
  void prepareVariables(const std::vector<const HTTParticle*>* legs, const std::vector<const HTTParticle*>* jets);
  void globalsHTT(const MLObjectMessenger* mess, const std::vector<const HTTParticle*>* legs, const HTTAnalysis::sysEffects* aSystEffect);
  void parseCfg(const std::string & cfgFileName);
  void fixSelections();
  void extractParticleProperties(const std::vector<const HTTParticle*>* legs,  const std::vector<const HTTParticle*>* jets);



};



#endif
