#include <map>

enum class PropertyEnum { PDGId = 0, 
charge = 1, 
decayMode = 2, 
discriminator = 3, 
muonID = 4, 
typeOfMuon = 5, 
byCombinedIsolationDeltaBetaCorrRaw3Hits = 6, 
byIsolationMVArun2v1DBoldDMwLTraw = 7, 
againstElectronMVA5category = 8, 
dxy = 9, 
dz = 10, 
SIP = 11, 
tauID = 12, 
combreliso = 13, 
leadChargedParticlePt = 14, 
isGoodTriggerType = 15, 
FilterFired = 16, 
L3FilterFired = 17, 
L3FilterFiredLast = 18, 
mc_match = 19, 
rawPt = 20, 
area = 21, 
PUJetID = 22, 
jecUnc = 23, 
Flavour = 24, 
bDiscriminator = 25, 
bCSVscore = 26, 
PFjetID = 27, 
NONE = 28
};

// Defined in PropertyEnum.cc
class PropertyEnumString
{
	public:
    	static const std::map<std::string, PropertyEnum> enumMap;
};

