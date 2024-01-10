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








# create learning layer, primary for the hidden layer
def create_learning_rule(l, lp =128):
    # initialize input-hidden layer weights
    # ini_weights = []
    # for i in range(len(l)-1):
    #     ini_weights.append(np.random.rand((l[i+1],l[i])))
    
    # these params denote the time inpulse
    top_x1TimeConstant = 10
    top_y1TimeConstant = 10

    hid_x1TimeConstant = 10
    hid_y1TimeConstant = 10
    
    #hyper-paras for weigts initilization
    mu =0.5
    var =0.02
    # update every sample 
    t_epoch =1 
    
    # the formula of the learning rate is 2^-lr
    lr =4 
    lrt = lr + 1  # top layer learning rate
    biasEx = 0
    
    lp = 7
    
    dw_top = '2^-' + str(lrt) + '*u' +str(lp) + '*y1*x1 - 2^-' + str(lrt + 1) + '*u'+str(lp)  + '*x1'
    # weight dynamics for the hidden layer
    dw_hidden = '2^-' + str(lr) + '*u' +str(lp) + '*y1*x1 - 2^-' + str(lr + 1) + '*u'+str(lp)  + '*x1' 
    # weight dynamics for the hidden layer


    emstdp_lr_top = Loihi2FLearningRule (
                    dt = '',
                    dw=dw_top,
                    x1_impulse=16,
                    x1_tau =top_x1TimeConstant,
                    y1_impulse=16,
                    y1_tau=top_y1TimeConstant,
                    t_epoch=1)    

    emstdp_lr_hidden = Loihi2FLearningRule (
                    dw=dw_hidden,
                    x1_impulse=16,
                    x1_tau=hid_x1TimeConstant,
                    y1_impulse=16,
                    y1_tau=hid_y1TimeConstant,
                    t_epoch=1
                    )   

    # try stdp
    stdp = STDPLoihi(learning_rate=1,
                 A_plus=1,
                 A_minus=-1,
                 tau_plus=10,
                 tau_minus=10,
                 t_epoch=4)
    scale = 1
    GCtoECDelayDeriv = int(62)
    # self.ECtoGCDelayDeriv = int(2)
    wtadelay = int(0)
    # self.lastECdelay = int(0)
    voldcy = int(0)
    curdcy = int(4095)
    
    hidden_biasMn, hidden_thold, out_biasMn, out_thold, ethold = calculate_thresholds(l)


    input_layer =LIFReset(
        shape = (l[0],),
        du = 1,
        dv = voldcy,
        reset_interval = 128,
        vth = 100
        
    )
    
    #hidden layer:
    hidden_layer = LearningLIF(
        shape = (l[1],),
        du = curdcy,
        dv = voldcy,
        bias_mant =hidden_biasMn,
        vth = hidden_thold,
        bias_exp = biasEx
    )

    out_layer = LearningLIF(
        shape = (l[2],),
        du = curdcy,
        dv = voldcy,
        bias_mant = out_biasMn,
        vth =out_thold,
        bias_exp = biasEx
        
    )


    # connection input_hidden
    input_hidden_conn = LearningDense(
    weights= to_integer(np.random.normal(mu,var,(l[1],l[0])),8),
    learning_rule= stdp,
    weight_exp=0,
    # sign_mode=SignMode.MIXED,
    # num_weight_bits=8,
    )

    # connection input_hidden
    hidden_out_conn = LearningDense(
    weights=  to_integer(np.random.normal(mu,var,(l[-1],l[1])),8),
    learning_rule=  stdp,
    # weight_exp=0,
    # sign_mode=SignMode.MIXED,
    # num_weight_bits=8,
    )
    

    # connect feed forward part:the major learning part
    input_layer.s_out.connect(input_hidden_conn.s_in)
    input_hidden_conn.a_out.connect(hidden_layer.a_in)
    hidden_layer.s_out.connect(hidden_out_conn.s_in)
    hidden_out_conn.a_out.connect(out_layer.a_in)

    # # #back propagation connection
    hidden_layer.s_out.connect(input_hidden_conn.s_in_bap)
    out_layer.s_out.connect(hidden_out_conn.s_in_bap)
    
    
    return input_layer, hidden_layer, out_layer

''''
run the experiment sample by sample
'''

def run_exp(data,labels,l = [200,100,10], num_steps =128):
    mu = 0.5
    var =0.2
    hid_wgt = 100*np.random.normal(mu,var,(l[1],l[0]))//3
    out_wgt = 100*np.random.normal(mu,var,(l[-1],l[1]))//3
    print("length of the dataset", len(data))
    for i in range(len(data)):
        #create EMSTDP instance
        network = EMSTDP(l,128,hid_wgt,out_wgt)
        hid_wgt, out_wgt = network.fit(data[i],labels[i],l,num_steps)
        #delete EMSTDP instance
        if i < len(data)-1:
            del network

    return hid_wgt,out_wgt

hid_wgt = np.array([[10,20],[10,20]])

''''
EMSTDP algorithm
'''    
class EMSTDP:
    def __init__(self, l, lp, hid_wgt, out_wgt, firing_rate = 1,scale =5,vth= 240):
        self.lp = lp
        self.reset_time = lp
        self.fr = firing_rate
        self.input_nodes = []
        self.hidden_nodes = []
        self.out_put = []
        # forward connections
        self.input_hid_conn = []
        self.hid_out_conn = []
        self.scale = scale

        # used for tracking
        self.out_spikes =[]
        self.hidden_spikes = []
        self.dummy_spikes = []
        self.neg_out_spikes = []
        self.neg_hidden_spikes = []
        self.label_spikes = []
        self.pos_hidden_spikes = []


        self.out_probe = Monitor()
        self.hidden_probe = Monitor()
        self.dummy_probe = Monitor()
        self.pos_hidden_probe = Monitor()
        self.neg_hidden_probe = Monitor()
        self.label_probe = Monitor()
        self.neg_out_probe = Monitor()
        # l input shape
        self.l = l
        # error path layers
        self.train = True
        self.error_out = []
        self.error_hidden = []
        self.neg_hidden =[]
        self.neg_out = []
        self.pos_hidden = []
        self.label = []
        self.dummy = []

        #pos path connections:
        self.label_out_conn = []
        self.label_hidden_conn = []
        self.pos_hidden_hidden_conn =[]

        #neg path connections
        self.dummy_conn = []
        self.neg_hidden_conn =[]
        self.neg_out_conn = []
        
        # alternative error path of output layer
        self.er_out_alt = []
        # label layer, constantly postive ones
        self.input_hidden_conn = []
        self.hidden_out_conn = []

        #firing rate of the input pixel
        self.fr = firing_rate

        self.vth = vth

        self.l = l



        self.w_h = []
        self.w_o = []
        self.e_h = []
        self.e_o = []
        self.threshold_h = []
        self.threshold_o = []
        self.ethreshold_h = []
        self.ethreshold_o = []


    
        hiddenThr1 = 0.5
        outputThr1 = 0.1
        threshold_h = hiddenThr1*self.fr
        threshold_o =  outputThr1*self.fr
        
        hiddens = [l[1]]

        inputs = l[0]

        outputs = l[-1]

        init=0

        dfa = 1

        w_h, self.w_o = Init_ForwardWgt(inputs, outputs, hiddens, init)


        self.w_h = np.array(w_h[0])

        self.e_h, self.e_o = Init_FeedbackWgt(inputs, outputs, hiddens, dfa=0)

       

        self.threshold_h, self.threshold_o, self.ethreshold_h, self.ethreshold_o = Init_Threshold(inputs,
                                                                                                  outputs, hiddens, threshold_h, 
                                                                                                  threshold_o,0)

        print(self.threshold_h)
        self.forward_network(l,hid_wgt,out_wgt)
        # self.create_pos_path(l)
        # self.create_neg_path(l)

    

    '''
    Feed-forward network,
    Including plastic connections
    '''        
    def forward_network(self,l,hid_wgt,out_wgt):
    

        self.input_nodes = LIFReset(
                    shape = (l[0],),
                    du = 1,
                    dv = 0,
                    reset_interval = 128,
                    bias_mant = 0,
                    name = 'input_nodes',
                    vth = self.scale
        )

        self.hidden_nodes = LIFReset(
            shape = (l[1],),
            du = 1,
            dv = 0,
            reset_interval = 128,
            # vth =  int(self.scale*self.threshold_h[0]),
            vth = 600,
            # vth = 100,
            # vth = self.scale,
            name = 'hidden_nodes'
        )

        self.out_put = LIFReset(
            shape = (l[2],),
            du = 1,
            dv = 0,
            reset_interval = 128,
            # vth = int(self.scale*self.threshold_o),
            vth = int(self.scale),
            # vth =  self.scale,
            name = 'out_spikes'
        )
                            
        #set up learning rules
        lr =4
        lrt = lr+1
        lp = 7
    
        dw_top = '2^-' + str(lrt) + '*u' +str(lp) + '*y1*x1 - 2^-' + str(lrt + 1) + '*u'+str(lp)  + '*x1'
        # weight dynamics for the hidden layer
        dw_hidden = '2^-' + str(lr) + '*u' +str(lp) + '*y1*x1 - 2^-' + str(lr + 1) + '*u'+str(lp)  + '*x1' 
        # weight dynamics for the hidden layer
        top_x1TimeConstant = 1
        top_y1TimeConstant = 1
    
        hid_x1TimeConstant = 1
        hid_y1TimeConstant = 1
    
        emstdp_lr_top = Loihi2FLearningRule (
                        dw=dw_top,
                        x1_impulse=1,
                        x1_tau =top_x1TimeConstant,
                        y1_impulse=1,
                        y1_tau=top_y1TimeConstant,
                        t_epoch=1)    
    
        emstdp_lr_hidden = Loihi2FLearningRule (
                        dw=dw_hidden,
                        x1_impulse=1,
                        x1_tau=hid_x1TimeConstant,
                        y1_impulse=1,
                        y1_tau=hid_y1TimeConstant,
                        t_epoch=1
                      )
        # hyper parameters for weight initialization
        mu =0.5
        var =0.4

        self.input_hid_conn = LearningDense(
                        # weights= (self.scale*self.w_h.T).astype(int),
                        weights = (self.scale*np.random.rand(l[1],l[0])).astype(int),
                        # weights = hid_wgt,
                        learning_rule= emstdp_lr_hidden,
                        name = 'input_hid_conn'
                        )

        self.hid_out_conn = LearningDense(
                        # weights= self.scale*self.w_o.T,
                        weights = (self.scale*np.random.rand(l[-1],l[1])).astype(int),
                        learning_rule= emstdp_lr_top,
                        name = 'hid_out_conn'
                        )
        # forward connection
        self.input_nodes.s_out.connect(self.input_hid_conn.s_in)
        self.input_hid_conn.a_out.connect(self.hidden_nodes.a_in)
        self.hidden_nodes.s_out.connect(self.hid_out_conn.s_in)
        self.hid_out_conn.a_out.connect(self.out_put.a_in)

        # # potential back propagation signals:
        # self.hidden_nodes.s_out.connect(self.input_hid_conn.s_in_bap)
        # self.out_put.s_out.connect(self.hid_out_conn.s_in_bap)



        
    '''
     label node always provides strong positive signals
    '''
    def create_pos_path(self, l):
        pos_scale = 10 
        self.label =LIFReset(
            shape = (l[-1],),
            du = 1,
            dv =0,
            reset_interval = 2,
            name = 'label',
            vth = self.scale*2 -1# any number that is greater than 3 is ok
        )
        self.pos_hidden = LIFReset(
            shape = (l[-2],),
            du =1,
            dv =0,
            reset_interval = 2,
            name = 'pos_hidden',
            vth = self.scale*2 -1 # any number that is greater than 3 is ok
        )
        
        
        # label connects to out layer: one-to-one connection
        self.label_out_conn = Dense(
            weights = self.scale*np.eye(l[-1]),
            name = 'label_out_conn'
        )
        self.label.s_out.connect(self.label_out_conn.s_in)
        self.label_out_conn.a_out.connect(self.out_put.a_in)
        
        #label to the postive penultimate layer,
        self.label_hidden_conn =Dense(
            weights =self.scale*np.random.rand(l[1],l[-1]),
            name = "label_hidden_conn"
        )
        
        self.pos_hidden_hidden_conn = Dense(
            weights = self.scale*self.threshold_h[0]*np.eye(l[-2]),
            name = "pos_hidden_hidden_conn"
        )
        
        self.label.s_out.connect(self.label_hidden_conn.s_in)
        self.label_hidden_conn.a_out.connect(self.pos_hidden.a_in)
        self.pos_hidden.s_out.connect(self.pos_hidden_hidden_conn.s_in)
        self.pos_hidden_hidden_conn.a_out.connect(self.hidden_nodes.a_in)

        

        
        
    #Other spikes in the final layer will provide negative signals down to the penultimate layer
    '''
    Negative signals provide 
    '''
    def create_neg_path(self,l):
        neg_scale = -10
        # first one to one from outlayer to the neg_out layer 
        self.neg_out = LIFReset(
            shape = (l[-1],),
            du =1,
            dv =0,
            name = 'neg_out',
            reset_interval = 2,
            vth = self.scale
        )
        self.neg_hidden = LIFReset(
            shape = (l[-2],),
            du =1,
            dv =0,
            reset_interval = 2,
            name ='neg_hidden',
            vth = self.scale
        )

        # need a dummy layer to recorde the negtive error path
        self.dummy = LIFReset(
            shape = (l[-1],),
            du =1,
            dv =0,
            reset_interval = 2,
            name = 'dummy',
            vth = 9 # any number that greater than 3 is ok
        )
        self.dummy_conn = Dense(
            weights = 10*np.eye(l[-1]),
            name = 'dummy_conn'
        )
        self.out_put.s_out.connect(self.dummy_conn.s_in)
        self.dummy_conn.a_out.connect(self.neg_out.a_in)
        
        
        #top down connection of the error path
        self.neg_conn = Dense(
            weights = self.scale*np.random.rand(l[1],l[-1]),
            name = 'neg_conn'
            
        )
        self.neg_out.s_out.connect(self.neg_conn.s_in)
        self.neg_conn.a_out.connect(self.neg_hidden.a_in)
        
        #One to one connection from error path to feed-forward path
        self.neg_out_conn = Dense(
            weights = -self.scale*self.threshold_o*np.eye(l[-1])//4,
            
        )
        self.neg_out.s_out.connect(self.neg_out_conn.s_in)
        self.neg_out_conn.a_out.connect(self.out_put.a_in)
        
        # hidden layer one to one connection
        self.neg_hidden_conn = Dense(
            weights = -self.scale*self.threshold_h[0]*np.eye(l[-2])//4,
            name = "neg_hidden_conn"
        )
        self.neg_hidden.s_out.connect(self.neg_hidden_conn.s_in)
        self.neg_hidden_conn.a_out.connect(self.hidden_nodes.a_in)
        
     #set up learning variables such that there're no postive or negitve signals      
    def set_training_para(self,l,train,label): 
         # Phase 2, the training phase
         if train:
             self.label.bias_mant.set(10*label)
             self.label.bias_exp.set(np.zeros(l[-1],))
             #integrate and fire neuron for neg_out 
             self.neg_out.dv.set(np.zeros(l[-1],))
         # In the non-training phase, the dummy layer will not produce any negative signals
         # The label will not produce any positive signals
         else:
             self.neg_out.dv.set(np.ones(l[-1],))
             self.label.bias_mant.set(np.zeros((l[-1]),))
    
    # transfer input image digits into bias
    # images are normalized to [0,1],1 d
    # please avoid using bias in lava-0.6
    def set_input_para(self,l,img):
        #input vth
        img = 0.5*self.scale*img
        bias = img.astype(int)
        self.input_nodes.bias_mant.set(bias)

    def create_probes(self,lp):
        '''
        Mainly the spikes of the hidden layer, output layer
        And the spikes of error paths, the weights of the plastic connections
        '''

        #track probe
        self.out_probe.probe(self.out_put.s_out,lp)
        self.hidden_probe.probe(self.hidden_nodes.s_out, lp) 
        self.dummy_probe.probe(self.dummy.s_out, lp) 
        self.pos_hidden_probe.probe(self.pos_hidden.s_out, lp) 
        self.neg_hidden_probe.probe(self.neg_hidden.s_out,lp)
        self.neg_out_probe.probe(self.neg_out.s_out,lp)
        self.label_probe.probe(self.label.s_out, lp)

        #return data
        self.out_spikes = self.out_probe.get_data()['out_put']['s_out']
        self.hidden_spikes = self.hidden_probe.get_data()['hidden_nodes']['s_out']
        self.dummy_spikes = self.dummy_probe.get_data()['dummy']['s_out']
        self.neg_out_spikes = self.neg_out_probe.get_data()['neg_out']['s_out']
        self.neg_hidden_spikes = self.neg_hidden_probe.get_data()['neg_hidden']['s_out']
        self.label_spikes = self.label_probe.get_data()['label']['s_out']
        self.pos_hidden_spikes =self.pos_hidden_probe.get_data()['pos_hidden']['s_out']
        
        return self.out_spikes, self.hidden_spikes, self.dummy_spikes,  self.neg_out_spikes,  self.neg_hidden_spikes,self.label_spikes,self.pos_hidden_spikes

    def run_network(self,num_steps = 10):
        '''
        For testing usage, run an image period
        '''
        # neg_out_prob = Monitor()
        # neg_out_prob.probe(self.neg_out.s_out,257)

        input_prob = Monitor()
        input_prob.probe(self.input_nodes.s_out,257)        

        output_prob = Monitor()
        output_prob.probe(self.out_put.s_out,257)
        
        self.input_nodes.run(condition=RunSteps(num_steps= 128), run_cfg=Loihi2HwCfg(select_tag=SELECT_TAG))
        # neg_out_spikes = neg_out_prob.get_data()['neg_out']['s_out']
        out_spikes = output_prob.get_data()['out_spikes']['s_out']
        input_spikes = input_prob.get_data()['input_nodes']['s_out']

        self.input_nodes.stop()

        return out_spikes, input_spikes
    '''
    Define the weight probation for the 
    '''
    def weight_probe(self,imgs,lp = 128):
        num_steps = len(imgs)*lp
        hidden_probe = Monitor()
        out_probe = Monitor()

        hidden_probe.probe(self.input_hid_conn.weights,1+num_steps)
        out_probe.probe(self.hid_out_conn.weights,1+num_steps)

        return hidden_probe,out_probe
        
    def prediction(self,pred):
        result = np.argmax(np.sum(pre,-1))
        return result
    
    
    #Test 
    def test(self, l, img, hid_wgt, out_wgt ,num_steps=64):

        input_nodes = LIFReset(
                    name = 'input',
                    shape = (l[0],),
                    du = 1,
                    dv = 0,
                    reset_interval = num_steps,
                    bias_mant = 0,
                    vth = self.scale

        )
        
        hidden_nodes = LIFReset(


            shape = (l[1],),
            du = 1,
            dv = 0,
            reset_interval = num_steps,
            vth =  int(self.scale*self.threshold_h[0]),
            name = 'hidden'
        )
        
        out_put = LIFReset(
            shape = (l[2],),
            du = 1,
            dv = 0,
            reset_interval = num_steps,
            vth = int(self.scale*self.threshold_o),
            # vth =  200,
            name = 'out'
        )
        
        input_hid =Dense(
           weights = hid_wgt
        )
        
        hid_out = Dense(
            weights = out_wgt
        )

        # Connections
        input_nodes.s_out.connect(input_hid.s_in)
        input_hid.a_out.connect(hidden_nodes.a_in)
        hidden_nodes.s_out.connect(hid_out.s_in)
        hid_out.a_out.connect(out_put.a_in)
        
        out_prob = Monitor()
        out_prob.probe(out_put.s_out, 1+len(img)*num_steps)
        input_nodes.run(condition=RunSteps(num_steps= 1), run_cfg=Loihi2HwCfg(select_tag=SELECT_TAG))
        for i in range(len(img)):
           input_nodes.bias_mant.set(img[i])
           input_nodes.run(condition=RunSteps(num_steps= num_steps), run_cfg=Loihi2SimCfg(select_tag=SELECT_TAG))

        spikes = out_prob.get_data()['out']['s_out']
        input_nodes.stop()

        '''
        spikes = [1+len(img)*time_steps,10]
        '''
        spikes = spikes[1:]
        print(spikes.shape)
        result = np.zeros((len(img),l[-1]))
        for i in range(len(img)):
            
            tmp = np.sum(spikes[i*(num_steps):(i+1)*num_steps,:],axis =0)
            result[i,:] = tmp
        
        return result
    
    def fit(self,img,label, l,num_steps=128):
        '''
        run for just one time step, for the bias setting purpose
        '''
        output_prob = Monitor()
        hidden_prob = Monitor()
        input_probe = Monitor()
        hidden_v_probe = Monitor()
        output_prob.probe(self.out_put.s_out,1 + len(img)*num_steps)
        hidden_prob.probe(self.hidden_nodes.s_out,1 + len(img)*num_steps)
        input_probe.probe(self.input_nodes.s_out,1 + len(img)*num_steps)
        hidden_v_probe.probe(self.hidden_nodes.v,1 + len(img)*num_steps)
        
        # hid_wgt,out_wgt = self.weight_probe(img)
        self.input_nodes.run(condition=RunSteps(num_steps= 1), run_cfg=Loihi2SimCfg(select_tag=SELECT_TAG))
        for i in range(len(img)):
            train = False
            self.set_input_para(l, img[i])
            self.set_training_para(l,train,labels[i])
            # run first 64 timesteps:
            for j in range(num_steps//2):
                self.input_nodes.run(condition=RunSteps(num_steps= 1), run_cfg=Loihi2SimCfg(select_tag=SELECT_TAG))
                self.neg_out.v.set(np.zeros((l[-1],)))
                self.label.v.set(np.zeros((l[-1],)))
            train =True

            # '''
            # now set up training variables, it doesn't need to change the voltage,
            # nothing here, run the second phase for EMSTDP
            # '''
            # # self.set_training_para(l,train,labels[i])
            # # self.label.bias_mant.set(self.scale*3*label[i])

            # self.input_nodes.run(condition=RunSteps(num_steps= num_steps//2), run_cfg=Loihi2SimCfg(select_tag=SELECT_TAG))
            self.input_nodes.run(condition=RunSteps(num_steps= num_steps), run_cfg=Loihi2SimCfg(select_tag=SELECT_TAG))
            '''
            Uncomment it if you need the dynamics of the weight.
            '''
            # hid_wgt.get_data()['input_hid_conn']['weights'][-1]
            # out_wgt.get_data()['hid_out_conn']['weights'][-1]

        # extract the weights for the final time step
        hid_wgt = self.input_hid_conn.weights.get()
        out_wgt = self.hid_out_conn.weights.get()
        
        #ouput labels
        out_spikes = output_prob.get_data()['out_spikes']['s_out']
        hidden_spikes = hidden_prob.get_data()['hidden_nodes']['s_out']
        input_spikes = input_probe.get_data()['input_nodes']['s_out']
        voltage = hidden_v_probe.get_data()['hidden_nodes']['v']
    
        
        self.input_nodes.stop()
        return hid_wgt, out_wgt ,out_spikes, hidden_spikes,input_spikes,voltage

