!SLURM=1 LOIHI_GEN=N3B3 PARTITION=oheogulch_2h
import os
import unittest
from unittest import skip
import numpy as np
import math
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.dense.process import LearningDense
from lava.proc.lif.process import LIFReset
from lava.utils.weightutils import SignMode

from lava.proc.dense.process import Dense

from lava.magma.core.run_configs import Loihi2HwCfg ,Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.callback_fx import NxSdkCallbackFx
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.proc.monitor.process import Monitor

from utils import accuracy
from utils import Init_ForwardWgt
from utils import Init_FeedbackWgt
from utils import Init_Threshold


import matplotlib.pyplot as plt
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
# input layer, hidden layer, output layer
l = [100,50,10]
# Set this tag to "fixed_pt" or "floating_pt" to choose the corresponding models.
SELECT_TAG = "fixed_pt"


# Plotting trace dynamics

def plot_time_series(time, time_series, ylabel, title):
    plt.figure(figsize=(10, 1))

    plt.step(time, time_series)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)

    plt.show()



# LIF parameters
if SELECT_TAG == "fixed_pt":
    du = 4095
    dv = 4095
elif SELECT_TAG == "floating_pt":
    du = 1
    dv = 1



def to_integer(weights,  bitwidth, normalize=True):
    """Convert weights and biases to integers.

    :param np.ndarray weights: 2D or 4D weight tensor.
    :param np.ndarray biases: 1D bias vector.
    :param int bitwidth: Number of bits for integer conversion.
    :param bool normalize: Whether to normalize weights and biases by the
        common maximum before quantizing.

    :return: The quantized weights and biases.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    max_val = np.max(np.abs(np.concatenate([weights], None))) \
        if normalize else 1
    a_min = -2**bitwidth
    a_max = - a_min - 1
    weights = np.clip(weights / max_val * a_max, a_min, a_max).astype(int)
    return weights
    
    
    
def init_wgts(wmin, wmax, wdes, wsrc, sd):
    # np.random.seed((sd+1)*10)

    tmpp = np.random.normal(0, np.sqrt(3.0 / float(wsrc)), [wdes, wsrc])

    amx = np.max(tmpp)
    amn = np.min(tmpp)
    a1 = (tmpp - amn) / (amx - amn)
    a1 = a1 * (wmax - wmin) + wmin
    wgts = np.clip(a1, a_min=wmin, a_max=wmax)

    # wgts = np.random.randint(low=wmin, high=wmax, size=(wdes, wsrc))

    wgts = wgts.astype(int)

    return wgts

def xavier_initialization(fan_in, fan_out, min, max):
    scale = (fan_in+fan_out)/fan_in
    limit = math.sqrt(3.0 * scale)
    weights = np.random.uniform(-limit, limit, size=(fan_out,fan_in))
    
    return weights
    
  
    
    
    


# initilalize threshold 
def init_th(wsrc, scale, wmax):
    hThr = float(scale / (1)) * wmax * wsrc * (np.sqrt(3.0 / float(wsrc)) / (2.0))
    hThr = int(hThr)
    return hThr


#calculating the error path threshold and threshold for the layers
# note that l is a 3- layer network
def calculate_thresholds(l):
    
    hiddens = [l[-2],l[-1]]
    classifier_vth = 0.3
    label_biasMn =0
    input_biasMn =0
    input_vth = 0.5  # first layer threshold
    LabeltoECwgt = 8
    for patternIdx in range(len(hiddens)):
        # hidden layer
        if patternIdx == len(hiddens) - 1:
            wsrc = hiddens[patternIdx - 1]
            hidden_thold = init_th(wsrc, classifier_vth, 255)
            hidden_biasMn = int(hidden_thold / 10)
            ethold = int(LabeltoECwgt)
        #input layer
        elif patternIdx == 0:
            wsrc = 200
            out_thold = init_th(wsrc,input_vth, 255)
            out_biasMn = int(out_thold / 10)
            ref = 1
            ethold = int(LabeltoECwgt)

    return hidden_biasMn, hidden_thold, out_biasMn, out_thold, ethold


import numpy as np
# !SLURM=1 LOIHI_GEN=N3B3 PARTITION=oheogulch_2h
import os
import numpy as np
import math
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.dense.process import LearningDense, Dense, DelayDense
from lava.proc.lif.process import LIFReset
from lava.utils.weightutils import SignMode

from lava.proc.dense.process import Dense

from lava.magma.core.run_configs import Loihi2HwCfg ,Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.callback_fx import NxSdkCallbackFx
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.proc.monitor.process import Monitor
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.proc.monitor.process import Monitor
import time



#convert fixed point paras to floating point settings
def converter(du,dv,vth):
    # # vth = int(vth*2**6)
    # du = (4095-du)*2**(-12)
    # dv = (4095-dv)*2**(-12)
    # vth = vth
    
    return du,dv,vth
    


def init_wgts(wmin, wmax, wdes, wsrc, sd):
    # np.random.seed((sd+1)*10)

    tmpp = np.random.normal(0, np.sqrt(3.0 / float(wsrc)), [wdes, wsrc])

    # wgts = tmpp * wmax

    # wm = np.max(tmpp)
    # wgts = tmpp * float(wmax/wm)
    # wgts = np.clip(wgts, -255, 255)

    amx = np.max(tmpp)
    amn = np.min(tmpp)
    a1 = (tmpp - amn) / (amx - amn)
    a1 = a1 * (wmax - wmin) + wmin
    wgts = np.clip(a1, a_min=wmin, a_max=wmax)

    # wgts = np.random.randint(low=wmin, high=wmax, size=(wdes, wsrc))

    wgts = wgts.astype(int)

    return wgts


def init_th(wsrc, layer, scale, wmax):
    hThr = float(scale/(1)) * wmax * wsrc * (np.sqrt(3.0 / float(wsrc)) / (2.0))
    # hThr = float(scale) * wmax * wsrc * (np.sqrt(3.0 / float(wsrc)) / (2.0))
    #     hThr = float(scale / (layer + 1)) * wmax * wsrc * 1.0 * 0.5
    hThr = int(hThr)
    return hThr


class emstdp:
    def __init__(self,
                 numInputs =200,
                 numHidNurns = [100,10],
                 num_steps = 128
                 ):
        # make sure the type of the parameters class
        # copy the parameters



        self.stim2bias = [int(1) for i in range(1)]
        self.stim2bias += [int(i * 1) for i in range(1, 256, 1)]
        self.train_data = []
        self.train_label = []
        self.numHidNurns =  numHidNurns
        self.numlayers = len(self.numHidNurns)
        self.numInputs = numInputs
        self.numMCs = self.numInputs
        self.numTargets = 10
        '''
        GC is the intermediate layer neurons
        '''
        self.numHidNurns = numHidNurns
        self.numGCs = np.sum(self.numHidNurns)


        # self.poswgtrng = 64
        # self.negwgtrng = -64

        # self.bposwgtrng = 128
        # self.bnegwgtrng = -128

        self.poswgtrng = 128
        self.negwgtrng = -128

        self.bposwgtrng = 255
        self.bnegwgtrng = -255

        
        self.inputs = []
        
        self.ec_pos =[[None],[None]]
        self.ec_neg = [[None],[None]]
        self.ec_tmp_pos =[[None],[None]]
        self.ec_tmp_neg = [[None],[None]]
        
        self.ec_pos_aux = [[None],[None]]
        self.ec_neg_aux = [[None],[None]]

        # probes related data structures
        self.allMCSomaProbes = None
        self.exc2InhConnProbes = None
        self.inh2ExcConnProbesPos = None
        self.inh2ExcConnProbesNeg = None
        self.mcADProbes = None
        self.mcSomaProbes = None
        self.gcProbes = None
        self.label = None
        self.numStepsRan = 0

        #weights for testing use
        self.hid_wgt = []
        self.out_wgt = []

        #testing usage
        self.test_inter = []
        self.test_out = []
        self.test_connections = []
        #probes
        self.out_probe = []

        
    def setupNetwork(self, train, wgt, bwgt):
        """ setups the EPL network """

        if train:
            self.trainbool = 1
        else:
            self.trainbool = 0

        self.allMCSomaGrp = None
        self.allLabelGrp = None
        self.wtaGrp = None

        self.allGCsPerPattern = dict()
        self.allPosECsPerPattern = dict()
        self.pos_ec_soma = None
        self.pos_ec_denrite = None
        self.allNegECsPerPattern = dict()
        self.neg_ec_soma = None
        self.neg_ec_dendrite = None
        self.allTmpPosECsPerPattern = dict()
        self.allTmpNegECsPerPattern = dict()

        self.forwardConns = dict()
        self.posbackwardConns = dict()
        self.negbackwardConns = dict()
        self.hiddens = dict()

        '''
        Create input patterns
        '''
        self.createMCNeurons()

        '''
        create label layer
        '''
        du,dv,vth = converter(4095,0,2)
        self.allLabelGrp = self.create_cx(-1,du,dv,vth)

        print("number of layers", self.numlayers)

        for patternIdx in range(self.numlayers):
            '''
            Hidden layers
            ''' 
            
            self.inhid_vth = 0.5 #0.3 # first layer threshold
            self.hid_vth = 0.3 # middle layer threshold
            self.classifier_vth = 0.3 # 0.5 # classifier layer threshold
            self.biasEx = 2 # bias exponential
            self.biasMn = 1 # bias mantissa default

            scale = 1
            self.GCtoECDelayDeriv = int(10)
            # self.ECtoGCDelayDeriv = int(2)
            self.wtadelay = int(0)
            # self.lastECdelay = int(0)
            self.voldcy = int(0)
            self.curdcy = int(4000)


            self.ECtoGCwgt = 255
            self.LabeltoECwgt = 8

            thold = 0
            wsrc = 0
            # calculating forward and error path thresholds
            if patternIdx == self.numlayers - 1:
                wsrc = self.numHidNurns[patternIdx - 1]
                thold = init_th(wsrc, patternIdx, self.classifier_vth, 255)
                self.biasMn = int(thold / 10)
                ref = 1
                ethold = int(self.LabeltoECwgt)
            elif patternIdx == 0:
                wsrc = self.numMCs
                thold = init_th(wsrc, patternIdx, self.inhid_vth, 255)
                self.biasMn = int(thold / 10)
                ref = 1
                if self.numlayers == 2:
                    ethold = int(self.LabeltoECwgt)
                else:
                    ethold = int(init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.inhid_vth, 255) / (patternIdx + 2))
                    # ethold = int(self.LabeltoECwgt) + (self.numlayers - patternIdx - 2)*ethold
            else:
                wsrc = self.numHidNurns[patternIdx - 1]
                thold = init_th(wsrc, patternIdx, self.hid_vth, 255)
                self.biasMn = int(thold / 10)
                ref = 1
                ethold = int(init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.hid_vth, 255) / (patternIdx + 2))
            
            if patternIdx != self.numlayers - 1:
                # self.allPosECsPerPattern[patternIdx] = self.net.createNeuronGroup()
                # self.allNegECsPerPattern[patternIdx] = self.net.createNeuronGroup()
                # self.allTmpPosECsPerPattern[patternIdx] = self.net.createCompartmentGroup()
                # self.allTmpNegECsPerPattern[patternIdx] = self.net.createCompartmentGroup()
                '''
                initialize posEC, negEC, tmp PosEC, tmp negEC
                posEC_soma: soma_du = 4095,soma_dv=0, bias_mant =0, vth =ethod
                pos_ec_denrite: du= 0, dv= 0, vth =2,
                
                negEC soma: du= 4095 ,dv=0 , vth = ehold,
                negEC dendrite: 
                '''
                # pos ec
                # self.allPosECsPerPattern[patternIdx] = andSoma(shape = self.numHidNurns[patternIdx])
                self.allPosECsPerPattern[patternIdx] = self.create_cx(patternIdx =patternIdx,du = 4095,dv = 4095,vth = 2)
                self.pos_ec_soma = self.create_cx(patternIdx =patternIdx, du=4095,dv=0, vth = ethold)
                self.pos_ec_dendrite = self.create_cx(patternIdx = patternIdx, du=0,dv=0,vth =2)
                # neg ec
                # self.allNegECsPerPattern[patternIdx] = andSoma(shape = self.numHidNurns[patternIdx])
                self.allNegECsPerPattern[patternIdx] = self.create_cx(patternIdx =patternIdx,du = 4095,dv = 4095,vth = 2)
                self.neg_ec_soma= self.create_cx(patternIdx = patternIdx, du=4095,dv=0, vth = ethold)
                self.neg_ec_dendrite = self.create_cx(patternIdx= patternIdx, du=0,dv=0,vth =2)
                

            else:
                '''
                du =0 , dv =0, vth =2,
                '''
                self.allPosECsPerPattern[patternIdx] = self.create_cx(patternIdx= patternIdx, du =4095, dv =0 ,vth =ethold)
                self.allNegECsPerPattern[patternIdx] = self.create_cx(patternIdx= patternIdx, du =4095, dv =0 ,vth =ethold)
                
            '''
            ec tmp pos: du =4095, dv =0, vthMant = 2
            ec tmp neg: du= 4095, dv =0, vthMant = 2
            '''
            self.allTmpPosECsPerPattern[patternIdx] = self.create_cx(patternIdx= patternIdx, du =4095,dv=0 ,vth =2)
            self.allTmpNegECsPerPattern[patternIdx] = self.create_cx(patternIdx= patternIdx, du =4095, dv=0,vth =2)
            #create GC neurons per pattern
            print("layer index", patternIdx)
            self.createGCNeuronsPerPattern(patternIdx)
            
        self.connectforwardConns(train, wgt)
        self.connectbackwardConns(bwgt)

    def create_cx(self,patternIdx, du,dv,vth,bias_mant =0, bias_exp =0):
        # du,dv,vth = converter(du,dv,vth)
        return LIFReset(shape = (self.numHidNurns[patternIdx],),
                   du = du,
                   dv = dv,
                   vth = vth,
                   bias_mant = bias_mant,
                   bias_exp = bias_exp,
                   reset_interval = 128
                   )
    def createTestGC(self, patternIdx):
                
            self.inhid_vth = 0.5 #0.3 # first layer threshold
            self.hid_vth = 0.3 # middle layer threshold
            self.classifier_vth = 0.3 # 0.5 # classifier layer threshold
            self.biasEx = 2 # bias exponential
            self.biasMn = 1 # bias mantissa default

            scale = 1
            self.GCtoECDelayDeriv = int(10)
            # self.ECtoGCDelayDeriv = int(2)
            self.wtadelay = int(0)
            # self.lastECdelay = int(0)
            self.voldcy = int(0)
            self.curdcy = int(4000)

            self.wtadelay = int(0)

            self.ECtoGCwgt = 255
            self.LabeltoECwgt = 8

            thold = 0
            wsrc = 0
            
            # calculating forward and error path thresholds
            if patternIdx == self.numlayers - 1:
                wsrc = self.numHidNurns[patternIdx - 1]
                thold = init_th(wsrc, patternIdx, self.classifier_vth, 255)
                self.biasMn = int(thold / 10)
                ref = 1
                ethold = int(self.LabeltoECwgt)
                print("ethreshold = ")
                print(ethold)
            elif patternIdx == 0:
                wsrc = self.numMCs
                thold = init_th(wsrc, patternIdx, self.inhid_vth, 255)
                self.biasMn = int(thold / 10)
                ref = 1
                if self.numlayers == 2:
                    ethold = int(self.LabeltoECwgt)
                else:
                    ethold = int(init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.inhid_vth, 255) / (patternIdx + 2))

            else:
                wsrc = self.numHidNurns[patternIdx - 1]
                thold = init_th(wsrc, patternIdx, self.hid_vth, 255)
                self.biasMn = int(thold / 10)
                ref = 1
                ethold = int(init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.hid_vth, 255) / (patternIdx + 2))
            
            
            du,dv,vth = converter(self.curdcy,self.voldcy,thold)
            self.hiddens[patternIdx] =self.create_cx(patternIdx = patternIdx,
                                                            du = du,dv=dv,vth = vth, bias_mant= self.biasMn,
                                                            bias_exp = self.biasEx)
            
    #create input neurons
    def createMCNeurons(self, biasMant=0):
        du =0
        dv =0
        vth =4*4
        # du,dv,vth = converter(du,dv,vth)
        '''
        input neurons
        '''
        mcSomaCx = LIFReset(
            shape = (self.numMCs,),
            du =du,
            dv = dv,
            vth = vth,
            reset_interval = 128,
            name = "train"
        )
        self.allMCSomaGrp = mcSomaCx
    #create input neurons
    def createTestInputs(self, biasMant=0):
        du =0
        dv =0
        vth =4*4
        # du,dv,vth = converter(du,dv,vth)
        '''
        input neurons
        '''
        test_mc = LIFReset(
            shape = (self.numMCs,),
            du =du,
            dv = dv,
            vth = vth,
            reset_interval = 128,
            name = "test"
        )
        self.inputs = test_mc
        
    def createGCNeuronsPerPattern(self, patternIdx):
            
        self.inhid_vth = 0.5 #0.3 # first layer threshold
        self.hid_vth = 0.3 # middle layer threshold
        self.classifier_vth = 0.3 # 0.5 # classifier layer threshold
        self.biasEx = 2 # bias exponential
        self.biasMn = 1 # bias mantissa default

        scale = 1
        self.GCtoECDelayDeriv = int(10)
        # self.ECtoGCDelayDeriv = int(2)
        self.wtadelay = int(0)
        # self.lastECdelay = int(0)
        self.voldcy = int(0)
        self.curdcy = int(4000)

        self.wtadelay = int(0)

        self.ECtoGCwgt = 255
        self.LabeltoECwgt = 8

        thold = 0
        wsrc = 0
        
        # calculating forward and error path thresholds
        if patternIdx == self.numlayers - 1:
            wsrc = self.numHidNurns[patternIdx - 1]
            thold = init_th(wsrc, patternIdx, self.classifier_vth, 255)
            self.biasMn = int(thold / 10)
            ref = 1
            ethold = int(self.LabeltoECwgt)
            print("ethreshold = ")
            print(ethold)
        elif patternIdx == 0:
            wsrc = self.numMCs
            thold = init_th(wsrc, patternIdx, self.inhid_vth, 255)
            self.biasMn = int(thold / 10)
            ref = 1
            if self.numlayers == 2:
                ethold = int(self.LabeltoECwgt)
            else:
                ethold = int(init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.inhid_vth, 255) / (patternIdx + 2))

        else:
            wsrc = self.numHidNurns[patternIdx - 1]
            thold = init_th(wsrc, patternIdx, self.hid_vth, 255)
            self.biasMn = int(thold / 10)
            ref = 1
            ethold = int(init_th(self.numHidNurns[patternIdx + 1], patternIdx, self.hid_vth, 255) / (patternIdx + 2))
        
        
        du,dv,vth = converter(self.curdcy,self.voldcy,thold)
        self.allGCsPerPattern[patternIdx] =self.create_cx(patternIdx = patternIdx,
                                                          du = du,dv=dv,vth = vth, bias_mant= self.biasMn,
                                                          bias_exp = self.biasEx)
        

        # creating connections from the error network to the forward path
        # hidden layer connections
        if patternIdx != self.numlayers - 1:
            
            posECtoTmpEC_conn = Dense(
                weights = 10*np.eye(self.numHidNurns[patternIdx])
            )

            '''
            to do connection 1: pos ec soma -> tmp pos ec
            weight = 10
            '''
            # pos_ec to tmp ec
            # src: pos ec soma
            # dst: tmp pos ec 
            self.allPosECsPerPattern[patternIdx].s_out.connect(posECtoTmpEC_conn.s_in)
            posECtoTmpEC_conn.a_out.connect(self.allTmpPosECsPerPattern[patternIdx].a_in)
            
            '''
            to do connection 2: neg ec soma-> tmp neg ec 
            weight = 10
            '''
            negECtoTmpEC_conn = Dense(
                weights = 10*np.eye(self.numHidNurns[patternIdx])
            )
            self.allNegECsPerPattern[patternIdx].s_out.connect(negECtoTmpEC_conn.s_in)
            negECtoTmpEC_conn.a_out.connect(self.allTmpNegECsPerPattern[patternIdx].a_in)
            
            '''
            to do connection 3: 
            src tmp pos EC-> dst gc(forward)
            weights: self.ECtoGCwgt
            weight_exp: ijexp
            '''
            tmpPosECtoGCconns = Dense(
                weights = self.ECtoGCwgt*(np.eye(self.numHidNurns[patternIdx]).astype(int))
            )
            self.allTmpPosECsPerPattern[patternIdx].s_out.connect(tmpPosECtoGCconns.s_in)
            tmpPosECtoGCconns.a_out.connect(self.allGCsPerPattern[patternIdx].a_in)
            
            '''
            to do connection 4:
            src tmp neg ec-> gc
            weight: -self.ECtoGCwgt
            weight_exp: ijexp
            '''
            tmpnegECtoGCConns = Dense (
                weights = -self.ECtoGCwgt*(np.eye(self.numHidNurns[patternIdx]).astype(int)))
            self.allTmpNegECsPerPattern[patternIdx].s_out.connect(tmpnegECtoGCConns.s_in)
            tmpnegECtoGCConns.a_out.connect(self.allGCsPerPattern[patternIdx].a_in)
            
            
            ##########################
            # creating connections from the forward path to auxilary error compartments to perform the derivative
            '''
            to do connection 5:
            using delay dense here:
            postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX, current remain a constant.
            weight = 10, np eye
            src GC -> dst pos EC dendrite[0]
            
            '''
            posGCtoECConns = DelayDense(
                weights = 10*np.eye(self.numHidNurns[patternIdx]).astype(int),
                delays = self.GCtoECDelayDeriv
            )
            
            self.allGCsPerPattern[patternIdx].s_out.connect(posGCtoECConns.s_in)
            posGCtoECConns.a_out.connect(self.pos_ec_dendrite.a_in)
            
            '''
            to do connection 6:
            DenseDelay
            weight =10,
            src GC -> dst neg EC dendrite[0]
            '''
            negGCtoECConns = DelayDense(
                weights = 10*np.eye(self.numHidNurns[patternIdx]).astype(int),
                delays = self.GCtoECDelayDeriv
            )
            self.allGCsPerPattern[patternIdx].s_out.connect(negGCtoECConns.s_in)
            negGCtoECConns.a_out.connect(self.neg_ec_dendrite.a_in)

        # for classifier layer
        if patternIdx == self.numlayers - 1:
            
          ########################################################
            # loss computation through spikes at the top layer of error path using connections from classifier and label
            labelscale = 1.5
            gcscale = 2

            
            '''
            to do connection 7
            weight: -int(self.LabeltoECwgt * gcscale), np eye
            src GC-> dst pos EC
            '''
            GCtoPosECConn = Dense(
                weights = -int(self.LabeltoECwgt * gcscale)*np.eye(self.numHidNurns[patternIdx])
            )
            self.allGCsPerPattern[patternIdx].s_out.connect(GCtoPosECConn.s_in)
            GCtoPosECConn.a_out.connect(self.allPosECsPerPattern[patternIdx].a_in)
            
            '''
            to do connection 8
            weights: int(self.LabeltoECwgt * labelscale), np eye
            src label -> dst pos ec
            '''
            LabeltoPosECConn = Dense(
                weights = int(self.LabeltoECwgt * labelscale)*np.eye(self.numHidNurns[patternIdx])
            )
            self.allLabelGrp.s_out.connect(LabeltoPosECConn.s_in)
            LabeltoPosECConn.a_out.connect(self.allPosECsPerPattern[patternIdx].a_in)
            '''
            to do connection 9:
            src GC -> dst neg EC
            weights: int(self.LabeltoECwgt * labelscale) np eye
            '''
            GCtoNegECConn = Dense(
                            weights = int(self.LabeltoECwgt * labelscale)*np.eye(self.numHidNurns[patternIdx])     
             )
            self.allGCsPerPattern[patternIdx].s_out.connect(GCtoNegECConn.s_in)
            GCtoNegECConn.a_out.connect(self.allNegECsPerPattern[patternIdx].a_in)

            '''
            to do connection 10:
            weights: -int(self.LabeltoECwgt * gcscale) np.eye(self.numHidNurns[patternIdx])
            src  label -> dst neg EC
            
            '''
            LabeltoNegECConn  = Dense(
                weights = -int(self.LabeltoECwgt * gcscale)*np.eye(self.numHidNurns[patternIdx])
            )
            self.allLabelGrp.s_out.connect(LabeltoNegECConn.s_in)
            LabeltoNegECConn.a_out.connect(self.allNegECsPerPattern[patternIdx].a_in)
            '''
            to do connection 11:
            weights: 10
            connMAT
            src pos EC -> dst tmp pos EC
            '''
            posECtoTmpPosEC = Dense(
                weights = int(10)*np.eye(self.numHidNurns[patternIdx])
            )
            self.allPosECsPerPattern[patternIdx].s_out.connect(posECtoTmpPosEC.s_in)
            posECtoTmpPosEC.a_out.connect(self.allTmpPosECsPerPattern[patternIdx].a_in)
            
            '''
            to do connection 12:
            weights: 10
            cnnMAT,
            src  neg EC -> dst tmp neg EC
            '''
            negECtoTmpNegEC = Dense(
                weights = int(10)*np.eye(self.numHidNurns[patternIdx])
            )
            self.allNegECsPerPattern[patternIdx].s_out.connect(negECtoTmpNegEC.s_in)
            negECtoTmpNegEC.a_out.connect(self.allTmpNegECsPerPattern[patternIdx].a_in)
            
            '''
            to do connection 13
            weights: int(self.ECtoGCwgt / 1) connMAT
            weight_exp: ijexp
            src tmp pos EC -> dst GC
            '''
            tmpPosECtoGC = Dense(
                weights =int(self.ECtoGCwgt / 1)*np.eye(self.numHidNurns[patternIdx])
            )
            self.allTmpPosECsPerPattern[patternIdx].s_out.connect(tmpPosECtoGC.s_in)
            tmpPosECtoGC.a_out.connect(self.allGCsPerPattern[patternIdx].a_in)
            '''
            to do connection 14:
            weights: -int(self.ECtoGCwgt / 1)
            weight_exp: ijexp
            src tmp neg EC -> dst GC
            '''
            tmpnegEctoGC = Dense(
                weights =-int(self.ECtoGCwgt / 1)*np.eye(self.numHidNurns[patternIdx])
            )
            self.allTmpNegECsPerPattern[patternIdx].s_out.connect(tmpnegEctoGC.s_in)
            tmpnegEctoGC.a_out.connect(self.allGCsPerPattern[patternIdx].a_in)

    def connectforwardConns(self, train, wgt):
        """ creates the GC->MC inhibitory connections for each pattern"""
        lr = 4  # 2^-lr learning rate
        lrt = lr + 1  # top layer learning rate
        lp = 7  # u7 -> 2^7 learning period

        top_x1TimeConstant = 64
        top_y1TimeConstant = 64

        hid_x1TimeConstant = 64
        hid_y1TimeConstant = 64

        dw_top = '2^-' + str(lrt) + '*u' + str(lp) + '*y1*x1 - 2^-' + str(lrt + 1) + '*u' + str(lp) + '*t*x1'
        dw_hid = '2^-' + str(lr) + '*u' + str(lp) + '*y1*x1 - 2^-' + str(lr + 1) + '*u' + str(lp) + '*t*x1'

        for pIdx in range(self.numlayers):
            if pIdx == self.numlayers - 1:
                # single update per sample
                # lr = self.net.createLearningRule(
                #     dd='2^0*x0 - 2^3*u7*d',
                #     dt='2^0*y0 - 2^0*u7*t',
                #     dw='2^-6*u7*y1*x1 - 2^-7*u7*t*x1', 
                #     x1Impulse=1,
                #     x1TimeConstant=32,
                #     y1Impulse=1,
                #     y1TimeConstant=128,
                #     y2Impulse=1,
                #     y2TimeConstant=4095,
                #     tEpoch=1)
                lr = Loihi2FLearningRule(
                    dd='2^0*x0 - 2^3*u7*d',
                    dt='2^0*y0 - 2^0*u7*t',
                    dw='2^-6*u7*y1*x1 - 2^-7*u7*t*x1',
                    x1_impulse = 1,
                    x1_tau = 32,
                    y1_impulse = 1,
                    y1_tau =128,
                    y2_impulse = 1,
                    y2_tau=4095,
                    t_epoch =1 
                )
                #dw='2^-5*u7*y1*x1 - 2^-6*u7*t*x1' for MSTAR works better
            else:
                # single update per sample
                lr = Loihi2FLearningRule(
                    dd='2^0*x0 - 2^3*u7*d',
                    dt='2^0*y0 - 2^0*u7*t',
                    dw='2^-5*u7*y1*x1 - 2^-6*u7*t*x1',
                    x1_impulse = 1,
                    x1_tau = 32,
                    y1_impulse = 1,
                    y1_tau =128,
                    y2_impulse = 1,
                    y2_tau=4000,
                    t_epoch=1)
            
            # if pIdx == self.numlayers - 1:
                # single update per sample
            #     lr = Loihi2FLearningRule(
            #         # dd='2^0*x0 - 2^3*u7*d',
            #         dt='2^0*y0 - 2^0*u7*t',
            #         dw=dw_top,  # add decay term
            #         x1_impulse=1,
            #         x1_tau=top_x1TimeConstant,
            #         y1_impulse=1,
            #         y1_tau=top_y1TimeConstant,
            #         x2_impulse=1,
            #         x2_tau=64,
            #         t_epoch=1)
            # elif pIdx == 0:
            #     # single update per sample
            #     lr = Loihi2FLearningRule(
            #         # dd='2^0*x0 - 2^3*u7*d',
            #         dt='2^0*y0 - 2^0*u7*t',
            #         dw=dw_hid,
            #         x1_impulse=1,
            #         x1_tau=hid_x1TimeConstant,
            #         y1_impulse=1,
            #         y1_tau=hid_y1TimeConstant,
            #         x2_impulse=1,
            #         x2_tau=64,
            #         t_epoch=1)
            # else:
            #     # single update per sample
            #     lr = Loihi2FLearningRule(
            #         # dd='2^0*x0 - 2^3*u7*d',
            #         dt='2^0*y0 - 2^0*u7*t',
            #         dw=dw_hid,
            #         x1_impulse=1,
            #         x1_tau=hid_x1TimeConstant,
            #         y1_impulse=1,
            #         y1_tau=hid_y1TimeConstant,
            #         x2_impulse=1,
            #         x2_tau=64,
            #         t_epoch=1)

            if pIdx == 0:
                # forWgts = np.ones((self.numHidNurns[pIdx], self.numMCs), int)*4
                if len(wgt) == 0:
                    forWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx], self.numMCs, pIdx)
                    # forWgts = np.random.randint(low=self.negwgtrng, high=self.poswgtrng, size=(self.numHidNurns[pIdx], self.numMCs), dtype=int)
                else:
                    forWgts = self.hid_wgt

                forConnGrp_1 = LearningDense(
                    weights = forWgts,
                    learning_rule= lr,
                    name = 'input_hidden_conn'
                    
                )

                self.forwardConns[pIdx] = forConnGrp_1
                self.allMCSomaGrp.s_out.connect(forConnGrp_1.s_in)
                forConnGrp_1.a_out.connect(self.allGCsPerPattern[pIdx].a_in)
                #backward connecions
                self.allGCsPerPattern[pIdx].s_out.connect(forConnGrp_1.s_in_bap)

            else:
                if len(wgt) == 0:
                    forWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx],
                                        self.numHidNurns[pIdx - 1], pIdx)
                else:
                    forWgts = self.out_wgt

                #                 print(forWgts)

                forConnGrp_2 = LearningDense(
                    weights = forWgts,
                    learning_rule= lr,
                    name = 'hid_out_conn'
                    
                )

                self.forwardConns[pIdx] = forConnGrp_2
                self.allGCsPerPattern[pIdx-1].s_out.connect(forConnGrp_2.s_in)
                forConnGrp_2.a_out.connect(self.allGCsPerPattern[pIdx].a_in)
                #backward connections for the second layer
                self.allGCsPerPattern[pIdx].s_out.connect(forConnGrp_2.s_in_bap)
                
    def connectbackwardConns(self, bwgt):
        # connections for the error path
        for pIdx in range(self.numlayers):

            if pIdx == 0:
                # self.backwardConns[pIdx] = self.net.createConnectionGroup()
                self.posbackwardConns[pIdx] = []
                self.negbackwardConns[pIdx] = []
                self.posbackwardConns[pIdx + 1] = []
                self.negbackwardConns[pIdx + 1] = []

            elif pIdx == self.numlayers - 1:

                if len(bwgt) == 0:
                    # posbackWgts = init_wgts(self.negwgtrng, self.poswgtrng, self.numHidNurns[pIdx-1], self.numHidNurns[pIdx])
                    posbackWgts = np.random.randint(low=self.bnegwgtrng, high=self.bposwgtrng,
                                                    size=(self.numHidNurns[pIdx - 1], self.numHidNurns[pIdx]),
                                                    dtype=int)
                else:
                    posbackWgts = bwgt[pIdx].T

                negbackWgts = - posbackWgts

                #error layer
                # src: pos ec ->pos ec soma
                # weights - posbackWgts
                posECtoSomaConn = Dense(
                    weights = posbackWgts
                )
                self.allPosECsPerPattern[pIdx].s_out.connect(posECtoSomaConn.s_in)
                posECtoSomaConn.a_out.connect(self.pos_ec_soma.a_in)

                #src: neg EC pidx -> dst: pos EC soma pidx -1
                # negbackwgts

                negECtpPosECsoma = Dense(
                   weights = negbackWgts
               )
                self.allNegECsPerPattern[pIdx].s_out.connect(negECtpPosECsoma.s_in)
                negECtpPosECsoma.a_out.connect(self.pos_ec_soma.a_in)

                # src: neg ec Pidx -> dst neg ec pidx -1 soma
                # weights = posbackWgts
                negECtonegECsoma = Dense(
                    weights = posbackWgts
                )
                self.allNegECsPerPattern[pIdx].s_out.connect(negECtonegECsoma.s_in)
                negECtonegECsoma.a_out.connect(self.neg_ec_soma.a_in)

                #src: posec pidx ->dst : negec soma 
                # weights negbacl wgts
                posECtonegsoma = Dense(weights = negbackWgts
                                       )
                self.allPosECsPerPattern[pIdx].s_out.connect(posECtonegsoma.s_in)
                posECtonegsoma.a_out.connect(self.neg_ec_soma.a_in)

                '''
                connect soma, dendrite to the final neuron
                weight =10
                '''
                pos_soma_conn = Dense(
                    weights = int(65)*np.eye(self.numHidNurns[pIdx-1])
                )
                pos_dendrite_conn = Dense(
                    weights = int(65)**np.eye(self.numHidNurns[pIdx-1])
                )

                self.pos_ec_soma.s_out.connect(pos_soma_conn.s_in)
                pos_soma_conn.a_out.connect(self.allPosECsPerPattern[pIdx-1].a_in)

                self.pos_ec_dendrite.s_out.connect(pos_dendrite_conn.s_in)
                pos_dendrite_conn.a_out.connect(self.allPosECsPerPattern[pIdx-1].a_in)

                neg_soma_conn = Dense(
                    weights = int(10)*np.eye(self.numHidNurns[pIdx-1])
                )
                neg_dendrite_conn = Dense(
                    weights = int(10)**np.eye(self.numHidNurns[pIdx-1])
                )

                self.neg_ec_soma.s_out.connect(neg_soma_conn.s_in)
                neg_soma_conn.a_out.connect(self.allNegECsPerPattern[pIdx-1].a_in)

                self.neg_ec_dendrite.s_out.connect(neg_dendrite_conn.s_in)
                neg_dendrite_conn.a_out.connect(self.allNegECsPerPattern[pIdx-1].a_in)


            else:
                '''
                more than two learning layers
                '''
                print("to do: more than 3 hidden layers")

    def idxToBases(self, inputList):
        """ maps the input data/sensor reading to an MC-AD bias current"""
        # inputList = list(inputList)
        # return [self.stim2bias[i] for i in inputList]
        return inputList

    def generate_data(self,train_data):
        '''
        train_data shape: N,200
        '''
        for i in range(len(train_data)):
            #arange train data
            self.train_data.append(self.idxToBases(train_data[0][i]))

    '''
    define probes for the network
    ''' 
    def probes(self,num_img=1):
        out_spikes = Monitor()
        out_spikes.probe(self.allGCsPerPattern[-1].s_out,self.num_steps*num_img)

        return out_spikes
    
    '''
    set up input data for each time step
    '''
    def set_input_data(self,img):
        #scale img to 0-255
        # train_data = self.idxToBases(img)
        #set up bias
        self.allLabelGrp.bias_mant.set(np.zeros((self.numTargets,)))
        #last EC layers voltages set to the negative number so that they won't fire during the first phase:
        # first layer
        self.allNegECsPerPattern[0].v.set(-7200*np.ones((self.numHidNurns[0],)))
        self.allPosECsPerPattern[0].v.set(-7200*np.ones((self.numHidNurns[0],)))
        # second layer
        self.allNegECsPerPattern[1].v.set(-7200*np.ones((self.numHidNurns[1],)))
        self.allPosECsPerPattern[1].v.set(-7200*np.ones((self.numHidNurns[1],)))
        #intermediate layer set
        self.allGCsPerPattern[0].v.set(np.zeros((self.numHidNurns[0],)))
        self.allGCsPerPattern[1].v.set(np.zeros((self.numHidNurns[1],)))


    
    #set up training parameters 
    def set_label_para(self,label):
        #set up label data to the label layer
        self.allLabelGrp.bias_mant.set(label)
        #reset EC layer v to 0
                #last EC layers voltages set to the negative number so that they won't fire during the first phase:
        # first layer
        self.allNegECsPerPattern[0].v.set(0*np.ones((self.numHidNurns[0],)))
        self.allPosECsPerPattern[0].v.set(0*np.ones((self.numHidNurns[0],)))
        # second layer
        self.allNegECsPerPattern[1].v.set(0*np.ones((self.numHidNurns[1],)))
        self.allPosECsPerPattern[1].v.set(0*np.ones((self.numHidNurns[1],)))


    '''
    set up another network to do the inference
    '''

    def test(self,imgs,hid_wgt = [] , out_wgt = [] ,num_steps = 128):
        #create connections
        #reconstruct input:
        self.createTestInputs()
        for patternIdx in range(self.numlayers):
            self.createTestGC(patternIdx= patternIdx)
        if len(hid_wgt) ==0:
            hid_weights = self.hid_wgt
            out_weights = self.out_wgt
        else:
            hid_weights = hid_wgt
            out_weights = out_wgt
            
        self.test_inter = Dense(
            weights = hid_weights
        )
        self.test_out = Dense(
            weights = out_weights
        )
        # first layer connection
        self.inputs.s_out.connect(self.test_inter.s_in)
        self.test_inter.a_out.connect(self.hiddens[0].a_in)
        # second layer connection
        self.hiddens[0].s_out.connect(self.test_out.s_in)
        self.test_out.a_out.connect(self.hiddens[1].a_in)

        #probes for the spikes
        out_probe = Monitor()
        out_probe.probe(self.hiddens[1].s_out, 1+len(imgs)*num_steps)
        self.inputs.run(condition=RunSteps(num_steps= 1), run_cfg=Loihi2HwCfg(
            select_tag = 'fixed_pt'
        ))
        # running test cases
        for i in range(len(imgs)):
            #converting img input to integer
            img = imgs[i]
            self.inputs.bias_mant.set(img)
            #running the network
            self.inputs.run(condition=RunSteps(num_steps= num_steps), run_cfg=Loihi2HwCfg(
            select_tag = 'fixed_pt'
        ))
            
        spikes = out_probe.get_data()[self.hiddens[1].name]['s_out']
        self.inputs.stop()
        result = np.zeros((len(imgs),10))
        for j in range(len(imgs)):
            tmp = np.sum(spikes[j*(num_steps):(j+1)*num_steps,:],axis =0)
            result[j,:] = tmp
        print(result)
        res = np.argmax(result,axis =-1)

        return res
        


    def fit(self,imgs,labels,num_steps=128):
        '''
        run for just one time step, for the bias setting purpose
        '''
        # hid_wgt,out_wgt = self.weight_probe(img)
        probe = Monitor()
        probe.probe(self.allGCsPerPattern[1].s_out,1+len(imgs)*128)
        self.allMCSomaGrp.run(condition=RunSteps(num_steps= 1), run_cfg=Loihi2HwCfg(
            select_tag = 'fixed_pt'
        ))
        '''
        set up probes
        '''
        for i in range(len(imgs)):
            train = False
            '''
            reset all forward layers intermediate values
            '''

            '''
            set up input data to bias
            set up label layer parameters
            '''
            img = imgs[i]
            self.allMCSomaGrp.bias_mant.set(img)
            self.set_input_data(img)
            # run first 64 timesteps:
            self.allMCSomaGrp.run(condition=RunSteps(num_steps= num_steps//2), run_cfg=Loihi2HwCfg(
                    select_tag = 'fixed_pt'))

            # self.allMCSomaGrp.bias_mant.set(np.zeros(200,))
            # '''
            # now set up training variables, it doesn't need to change the voltage,
            # nothing here, run the second phase for EMSTDP
            # '''
            '''
            set up bias
            '''
            # idx = labels[i]
            # label = np.array([ 0 for j in range(self.numTargets)])
            # label[idx] =1
            self.set_label_para(labels[i])

            self.allMCSomaGrp.run(condition=RunSteps(num_steps= num_steps//2), run_cfg=Loihi2HwCfg(select_tag= 'floating_pt'))
            '''
            Uncomment it if you need the dynamics of the weight.
            '''
            # hid_wgt.get_data()['input_hid_conn']['weights'][-1]
            # out_wgt.get_data()['hid_out_conn']['weights'][-1]
        self.hid_wgt = self.forwardConns[0].weights.get()
        self.out_wgt = self.forwardConns[1].weights.get()
        spikes = probe.get_data()[self.allGCsPerPattern[1].name]['s_out']
        self.allMCSomaGrp.stop()
        #print non zero elements
        #print(np.count_nonzero(spikes))
        spikes = spikes[1:]
        result = np.zeros((len(imgs),10))
        for j in range(len(imgs)):
            tmp = np.sum(spikes[j*(num_steps):(j+1)*num_steps,:],axis =0)
            result[j,:] = tmp
        print(result[-10:])
        res = np.argmax(result,axis =-1)
        #print the last 10 spiking results
        print(res[-10])
        # #saving weigt
        np.save("hid_wgt",self.hid_wgt)
        np.save("out_wgt",self.out_wgt)

        #return weights
    def weights(self):
        return self.hid_wgt, self.out_wgt 


#
