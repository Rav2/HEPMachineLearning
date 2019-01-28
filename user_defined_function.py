import numpy as np
import pandas as pd

#jest to jedyne miejsce w którym trzeba pisać coś zależnego od problemu. 

#don't shuffle data. The framework will do it for you
#return is_this_classification, numerical_features, cathegorical_features, 
# ig_numerical_features, ig_cathegorical_features 
# where is_this_classification jest true jeśli to jest 
# binarna klasyfikacja, dla regresji to jest false. 
#. 4 ostatnie argumenty to slowniki ktore nazwom featcherow
# przyporzadkowuja numpyowe tablice 2d, których pierwszy index
# to index przypadku, a drugi odpowiada za dlugosc featcheru. 
#kategoryczne rzeczy to inty zawsze, label takze jesli jest kategoryczny
def get_data():
    zrodlo = "data/htt_features_train.pkl"
    legs, jets, global_params, properties = pd.read_pickle(zrodlo)

    #properties = OrderedDict(sorted(properties.items(), key=lambda t: t[0]))

    print("no of legs: ", len(legs))
    print("no of jets: ", len(jets))
    print("global params: ", global_params.keys())
    print("object properties:",properties.keys())

    genMass = np.array(global_params["genMass"])
    fastMTT = np.array(global_params["fastMTTMass"])
    visMass = np.array(global_params["visMass"])
    caMass = np.array(global_params["caMass"])
    leg1P4 = np.array(legs[0])
    leg2P4 = np.array(legs[1])
    leg1GenP4 = np.array(legs[2])
    leg2GenP4 = np.array(legs[3])        
    leg2Properties = np.array(properties["leg_2_decayMode"])
    jet1P4 = np.array(jets[1])
    jet2P4 = np.array(jets[2])        
    met = np.array(jets[0][0:3])

    #we want vertical vectors
    genMass = np.reshape(genMass, (-1,1))
    visMass = np.reshape(visMass, (-1,1))
    caMass = np.reshape(caMass, (-1,1))
    fastMTT = np.reshape(fastMTT, (-1,1))
    leg2Properties = np.reshape(leg2Properties, (-1,1))

    #first index should be example index
    leg1P4 = np.transpose(leg1P4)
    leg2P4 = np.transpose(leg2P4)
    leg1GenP4 = np.transpose(leg1GenP4)
    leg2GenP4 = np.transpose(leg2GenP4)        
    jet1P4 = np.transpose(jet1P4)
    jet2P4 = np.transpose(jet2P4)
    met = np.transpose(met)
    
    #basic feature engeneering
    leg1Pt = np.sqrt(leg1P4[:,1]**2 + leg1P4[:,2]**2)
    leg2Pt = np.sqrt(leg2P4[:,1]**2 + leg2P4[:,2]**2)
    leg1Pt = np.reshape(leg1Pt, (-1,1))
    leg2Pt = np.reshape(leg2Pt, (-1,1)) 
    metMag = met[:,0]
    #smear met
    #why ?
    smearMET = False
    if smearMET:            
        sigma = 2.6036 -0.0769104*metMag + 0.00102558*metMag*metMag-4.96276e-06*metMag*metMag*metMag
        print("Smearing metX and metY with sigma =",sigma)
        metX = met[:,1]*(1.0 + sigma*np.random.randn(met.shape[0]))
        metY = met[:,2]*(1.0 + sigma*np.random.randn(met.shape[0]))
        metMag = np.sqrt(metX**2 + metY**2)
        met = np.stack((metMag, metX, metY), axis=1)

    metMag = np.reshape(metMag, (-1,1))    

    #leg2GenEnergy = leg2GenP4[:,0]
    #leg2GenEnergy = np.reshape(leg2GenEnergy, (-1,1))
    important_things = {"genMass": genMass, 
                        "fastMTT": fastMTT,
                        "caMass": caMass, 
                        "visMass": visMass, 
                        "leg1P4": leg1P4, 
                        "leg2P4": leg2P4, 
                        "met": met}
            
    #Select events with MET>10
    mask_1 = met[:,0]>10 
    #features = features[index]

    #select events with genMass < 250
    mask_2 = important_things["genMass"][:,0]<250 
    #features = features[index]

    #select events with genMass > 50
    mask_3 = important_things["genMass"][:,0]>50 
    #features = features[index]

    mask = np.logical_and(np.logical_and(mask_1, mask_2),
        mask_3)
    

    #Quantize the output variable into self.nLabelBins
    #or leave it as a floating point number 
    for k in important_things.keys():
        important_things[k] = important_things[k][mask]
               
    labels = important_things.pop("genMass", None) # genMass is label
    labels = np.reshape(labels, (-1,1))
    '''
    if self.nLabelBins>1:
        est = preprocessing.KBinsDiscretizer(n_bins=self.nLabelBins, encode='ordinal', strategy='uniform')
        tmp = np.reshape(features[:,0], (-1, 1))
        est.fit(tmp)
        labels = est.transform(tmp) + 1#Avoid bin number 0
    else:                
        labels = features[:,0]
    '''

    #Apply all transformations to fastMTT, caMass and visMass columns,
    #as we want to plot it, but remove those columns from the model features
    fastMTT = important_things.pop("fastMTT", None)
    caMass = important_things.pop("caMass", None)
    visMass = important_things.pop("visMass", None)
    features = important_things

    num_features = features 
    cat_features = {}
    ig_num_features = {
        "fastMTT": fastMTT,
        "caMass": caMass, 
        "visMass": visMass }
    ig_cat_features = {}
    #print("Input data shape:",features.shape)
    #print("Label bins:",self.nLabelBins, "labels shape:",labels.shape)
    #print("caMass.shape",caMass.shape)

    return False, num_features, cat_features, ig_num_features, \
            ig_cat_features, labels
