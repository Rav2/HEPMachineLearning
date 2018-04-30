#include "MLAnalyzer.h"
#include <utility>
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/ini_parser.hpp"
#include "boost/tokenizer.hpp"
#include "boost/functional/hash.hpp"


MLAnalyzer::MLAnalyzer(const std::string & aName, const std::string & aDecayMode = "None") :
myName_(aName)
{
	execution_counter_=0;
	MLTree_=NULL;
	cfgFileName = "ml_Properties.ini";
	try
	{
		parseCfg(cfgFileName);
	}
	catch(std::exception& e)
	{
		std::cerr<<"[ML][WARNING]\t PARTICLE PROPERTY PARSER FAILED! READ THE MESSAGE BELOW:"<<std::endl;
		std::cerr<<e.what()<<std::endl;
	}
}

MLAnalyzer::~MLAnalyzer()
{
}

void MLAnalyzer::parseCfg(const std::string & cfgFileName)
{
  boost::property_tree::ptree pt;
  boost::property_tree::ini_parser::read_ini(cfgFileName, pt);

  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep(", ");

  // loop over all possible properties
  for(auto it=PropertyEnumString::enumMap.begin(); it!=PropertyEnumString::enumMap.end(); ++it)
  {
  	std::string property = it->first;
  	boost::optional<std::string> str_opt = pt.get_optional<std::string>(std::string("MLAnalyzer.")+property);
  	if(!str_opt)
  	{
  		std::cerr<<"[ML][WARNING] PARAMETER: "<<property<<" NOT IN PARAM FILE!"<<std::endl;
  		continue;
  	}
  	std::string str = static_cast<std::string>(*str_opt);
  	tokenizer tokens(str, sep);
  	for (auto it: tokens) 
	{
		if("l"==it.substr(0,1))
		{
			auto iter = params_legs_.find(property);
			if (iter == params_legs_.end())
			{
				std::set<unsigned> s;
			    params_legs_[property]=s;
			}

			if(it.size()==1)
				params_legs_[property].insert(1000); // dummy value
			else
			{
				unsigned val = std::stoul(it.substr(1));
				params_legs_[property].insert(val);
			}
		}
		else if("j"==it.substr(0,1))
		{
			auto iter = params_jets_.find(property);
			if (iter == params_jets_.end())
			{
				std::set<unsigned> s;
			    params_jets_[property]=s;
			}

			if(it.size()==1)
				params_jets_[property].insert(1000); // dummy value
			else
			{
				unsigned val = std::stoul(it.substr(1));
				params_jets_[property].insert(val);
			}
		}
		else
			throw std::invalid_argument(std::string("[ERROR] WRONG SELECTOR FOR PARTICLE PROPERTY! PROPERTY: ")+\
				property+std::string(" SELECTOR: ")+it);
	}
  }
}

void MLAnalyzer::fixSelections()
{
	for (auto it=params_legs_.begin(); it!=params_legs_.end(); ++it)
	{
		auto search = (it->second).find(1000);
		if(search != (it->second).end())
		{
			(it->second).erase(search);
			for(unsigned ii=1; ii<=legs_no_; ii++)
			{
				(it->second).insert(ii);
			}
		}
	}
	for (auto it=params_jets_.begin(); it!=params_jets_.end(); ++it)
	{
		auto search = (it->second).find(1000);
		if(search != (it->second).end())
		{
			(it->second).erase(search);
			for(unsigned ii=1; ii<=legs_no_; ii++)
			{
				(it->second).insert(ii);
			}
		}
	}
}

void MLAnalyzer::addBranch(TTree *tree)
{

	if(!MLTree_)
	{
		// Will execute after creation of a MLAnalyzer object
		if(tree)
		{
			// Adding branches for global parameters and saving the address of TTree
			MLTree_ = tree;
			tree->Branch("visMass", &visMass_);
			tree->Branch("higgsMassTrans", &higgsMassTrans_);
			tree->Branch("higgsPT", &higgsPT_);
			tree->Branch("BJetBetaScore", &betaScore_);
			tree->Branch("nJets30", &nJets30_);
		}
	}
	else
	{
		// Will execute once during the first call of MLAnalyzer::analyze()
		std::string leg_s ("leg_");
		std::string jet_s ("jet_");
		std::string endings[4]={"_E", "_pX", "_pY", "_pZ"};
		try
		{
			for(unsigned ii=0; ii<legs_p4_.size();ii++)
			{
				std::string name = leg_s+std::to_string(ii/4+1)+endings[ii%4];
				tree->Branch(name.c_str(), &legs_p4_.at(ii));

			}
			for(unsigned ii=0; ii<jets_p4_.size();ii++)
			{
				std::string name = jet_s+std::to_string(ii/4+1)+endings[ii%4];
				tree->Branch(name.c_str(), &jets_p4_.at(ii));
			}

			
			for (auto it=params_legs_.begin(); it!=params_legs_.end(); ++it)
			{
				std::string name = it->first;
				std::set<unsigned> selected_legs = it->second;
				for(auto sel=selected_legs.begin(); sel!=selected_legs.end(); ++sel)
				{
					unsigned no = *sel;
					if(no > legs_no_ || no==0)
					{
						std::cerr<<"[ML][WARNING]\tRequest to add branch for params for leg with larger number than accessible (or zero). I will ignore the request."<<std::endl;
					}
					else
					{
						std::string full_name = leg_s+std::to_string(no)+std::string("_")+name;
						tree->Branch(full_name.c_str(), &(params_.at(std::string("legs_")+name).at(no-1)));
					}

				}
			}
			for (auto it=params_jets_.begin(); it!=params_jets_.end(); ++it)
			{
				std::string name = it->first;
				std::set<unsigned> selected_jets = it->second;
				for(auto sel=selected_jets.begin(); sel!=selected_jets.end(); ++sel)
				{
					unsigned no = *sel;
					if(no > jets_no_ || no==0)
					{
						std::cerr<<"[ML][WARNING]\tRequest to add branch for params for jet with larger number than accessible (or zero). I will ignore the request."<<std::endl;
					}
					else
					{
						std::string full_name = jet_s+std::to_string(no)+std::string("_")+name;
						tree->Branch(full_name.c_str(), &(params_.at(std::string("jets_")+name).at(no-1)));
					}

				}
			}
		}
		catch(const std::out_of_range& e)
		{
			std::throw_with_nested(std::runtime_error("[ERROR] OUT OF RANGE IN MLAnalyzer::addBranch!"));
		}
		catch(const std::exception& e)
		{
		     std::throw_with_nested(std::runtime_error("[ERROR] UNKNOWN ERROR IN MLAnalyzer::addBranch!"));
		} 

		}
}


void MLAnalyzer::prepareVariables(const std::vector<const HTTParticle*>* legs, const std::vector<const HTTParticle*>* jets)
{
	try
	{
		/*** FOUR-MOMENTA ***/
		for(unsigned leg=0; leg<legs->size();leg++)
			for(unsigned coord=0; coord<4; coord++)
			{
				double x = 0.0;
				legs_p4_.push_back(x);

			}
		for(unsigned jet=0; jet<jets->size();jet++)
			for(unsigned coord=0; coord<4; coord++)
			{
				double x = 0.0;
				jets_p4_.push_back(x);
			}
		/*** PARTICLE PROPERTIES ***/
		for (auto it=params_legs_.begin(); it!=params_legs_.end(); ++it)
		{
			std::string name = it->first;
			std::vector<double> v;
			for(unsigned leg=0; leg<legs->size(); leg++)
			{
				double x = 0.0;
				v.push_back(x);
			}
			params_[std::string("legs_")+name] = v;
		}
		for (auto it=params_jets_.begin(); it!=params_jets_.end(); ++it)
		{
			std::string name = it->first;
			std::vector<double> v;
			for(unsigned jet=0; jet<jets->size(); jet++)
			{
				double x = 0.0;
				v.push_back(x);
			}
			params_[std::string("jets_")+name] = v;
		}
	}
	catch(const std::out_of_range& e)
	{
		std::throw_with_nested(std::runtime_error("[ERROR] OUT OF RANGE IN MLAnalyzer::prepareVariables!"));
	}
	catch(const std::exception& e)
	{
	     std::throw_with_nested(std::runtime_error("[ERROR] UNKNOWN ERROR IN MLAnalyzer::prepareVariables!"));
	} 

}

void MLAnalyzer::extractP4(const TLorentzVector& v, const std::string destination, const unsigned no)
{
	try
	{
		if(destination=="jets")
		{
			jets_p4_.at(4*no) = v.T();
			jets_p4_.at(4*no+1) = v.X();
			jets_p4_.at(4*no+2) = v.Y();
			jets_p4_.at(4*no+3) = v.Z();
		}
		else if(destination=="legs")
		{
			legs_p4_.at(4*no) = v.T();
			legs_p4_.at(4*no+1) = v.X();
			legs_p4_.at(4*no+2) = v.Y();
			legs_p4_.at(4*no+3) = v.Z();
		}
	}
	catch(const std::out_of_range& e)
	{
		std::throw_with_nested(std::runtime_error(std::string("[ERROR] OUT OF RANGE IN MLAnalyzer::extractP4!\nJETS_VECTOR_SIZE: ")\
		 + std::to_string(jets_p4_.size()) +  std::string(", LEGS_VECTOR_SIZE: ") + std::to_string(jets_p4_.size()) + \
		 std::string(", TRYING TO ACCESS: [") + std::to_string(4*no) + std::string(",") + std::to_string(4*no+3)+ std::string("]!")));
	}
	catch(const std::exception& e)
	{
	     std::throw_with_nested(std::runtime_error("[ERROR] UNKNOWN ERROR IN MLAnalyzer::extractP4!"));
	} 
}

void MLAnalyzer::extractParticleProperties( const std::vector<const HTTParticle*>* legs,  const std::vector<const HTTParticle*>* jets)
{
	try
	{
		std::string legs_s ("legs_");
		std::string jets_s ("jets_"); 
		for (auto it=params_legs_.begin(); it!=params_legs_.end(); ++it)
		{
			std::string name = it->first;
			// std::cout<<"NAME: "<<name<<" ORIGINAL: "<<it->first<<std::endl;
			for(auto iter = (it->second).begin(); iter != (it->second).end(); ++iter)
			{
				unsigned no = *iter;
				// assign property value to variable in params_
				(params_.at(legs_s+name)).at(no-1) = (legs->at(no-1))->getProperty(PropertyEnumString::enumMap.at(name));
			}
		}
	}
	catch(const std::out_of_range& e)
	{
		std::throw_with_nested(std::runtime_error(std::string("[ERROR] OUT OF RANGE IN MLAnalyzer::extractParticleProperties")));
	}
	catch(const std::exception& e)
	{
	     std::throw_with_nested(std::runtime_error("[ERROR] UNKNOWN ERROR IN MLAnalyzer::extractParticleProperties!"));
	} 
		
}
void MLAnalyzer::globalsHTT(const MLObjectMessenger* mess, const std::vector<const HTTParticle*>* legs, const HTTAnalysis::sysEffects* aSystEffect)
{
	try
	{
		void* p = NULL;
		const HTTParticle* aMET = mess->getObject(static_cast<HTTParticle*>(p), std::string("amet"));
		const float* bs = mess->getObject(static_cast<float*>(p), "beta_score");
		const float* higgs_mass = mess->getObject(static_cast<float*>(p), "higgs_mass_trans");
		const int* nJets = mess ->getObject(static_cast<int*>(p), "nJets30");

		const HTTParticle* leg1 = legs->at(0);
		const HTTParticle* leg2 = legs->at(1);

		if(!(leg1 && leg2 && aMET && aSystEffect && bs && higgs_mass && nJets))
			throw std::logic_error("[ERROR] NULL POINTERS PRESENT!");
		const TLorentzVector & aVisSum = leg1->getP4(*aSystEffect) + leg2->getP4(*aSystEffect);
	    visMass_ = aVisSum.M();
	    higgsPT_ =  (aVisSum + aMET->getP4(*aSystEffect)).Pt();
	    betaScore_ = *bs;
	    higgsMassTrans_ = *higgs_mass;
	    nJets30_ = *nJets;
	}
	catch(const std::out_of_range& e)
	{
		std::throw_with_nested(std::runtime_error("[ERROR] CALCULATING GLOBAL PARAMETERS FOR HTT REQUIRES AT LEAST 2 LEGS!"));
	}
	catch(const std::logic_error& e)
	{
		std::throw_with_nested(std::runtime_error("[ERROR] CANNOT FILL FIELDS IN MLAnalyzer::globalsHTT!"));
	}
	catch(const std::exception& e)
	{
	     std::throw_with_nested(std::runtime_error("[ERROR] UNKNOWN ERROR IN MLAnalyzer::globalsHTT!"));
	} 
}


bool MLAnalyzer::analyze(const EventProxyBase& iEvent, ObjectMessenger *aMessenger)
{
	try
	{
		// std::cout<<(int)PropertyEnumString::enumMap.at(std::string("charge"))<<" "<<execution_counter_<<std::endl;

		if(aMessenger and std::string("MLObjectMessenger").compare(0,17,aMessenger->name()))
		{
			void* p = NULL;
			MLObjectMessenger* mess = ((MLObjectMessenger*)aMessenger);
			const std::vector<const HTTParticle*>* legs = mess->getObjectVector(static_cast<HTTParticle*>(p), std::string("legs"));
			const std::vector<const HTTParticle*>* jets = mess->getObjectVector(static_cast<HTTParticle*>(p), std::string("jets"));
			const HTTAnalysis::sysEffects* aSystEffect = mess->getObject(static_cast< HTTAnalysis::sysEffects*>(p), std::string("systEffect"));

			//TODO: uzyc parsera z boosta albo poszukac czy w std cos jest
			// parametry dla jetow i legow
			if(mess && aSystEffect && legs && jets)
			{
				if(execution_counter_ == 0)
				{
					// Number of leg and jet object must be constant! Objects can be dummy.
					legs_no_ = legs->size();
					jets_no_ = jets->size();
					// Knowing the number of jets and legs fix the selections to match it (remove 1000 dummy values)
					fixSelections();
					// Stuff to do at the first execution, e.g. assigning branches
					std::cout<<"[ML]\tAdding branches for ML analysis."<<std::endl;
					prepareVariables(legs, jets);
					addBranch(MLTree_);
					
				}
				if(legs->size()>0)
					execution_counter_++;

				if(jets->size() != jets_no_ || legs->size() != legs_no_)
					throw std::logic_error("[ERROR] MLAnalyzer REQUIRES CONSTANT NUMBER OF JETS AND LEGS IN EACH CALL!");
				// variable "no" ensures assigning values to correct leg/jet; it has to be set to 0 before for_each
				unsigned no = 0;
				std::for_each(legs->begin(), legs->end(), [&](const HTTParticle* leg)
				{
					extractP4(leg->getP4(*aSystEffect), std::string("legs"), no);
					no++;
				});
				no = 0;
				std::for_each(jets->begin(), jets->end(), [&](const HTTParticle* jet)
				{
					extractP4(jet->getP4(*aSystEffect), std::string("jets"), no);
				});

				// Extract properties defined in PropertyEnum.h
				extractParticleProperties(legs, jets);
				// Dealing with global parameters, one can put here another function for other analysis
				globalsHTT(const_cast<const MLObjectMessenger*>(mess), legs, aSystEffect);
				
				// Cleaning the content of ObjectMessenger NO DELETE IS CALLED!!!
				mess->clear();
			}

		}
	}
	catch(const std::exception& e)
	{
		std::throw_with_nested(std::runtime_error("[ERROR] UNKNOWN ERROR IN MLAnalyzer::analyze!"));
	}
	return true;
}

bool MLAnalyzer::analyze(const EventProxyBase& iEvent)
{
	return analyze(iEvent, NULL);
}



