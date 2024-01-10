import matplotlib.pyplot as plt
import numpy as np
from lava.proc.lif.process import LearningLIF
from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat, LearningNeuronModelFixed
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag



def plot_spikes(spikes, figsize, legend, colors, title, num_steps):
    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps + 1, 25)

    plt.figure(figsize=figsize)

    plt.eventplot(positions=spikes,
                  lineoffsets=offsets,
                  linelength=0.9,
                  colors=colors)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")

    plt.xticks(num_x_ticks)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.yticks(ticks=offsets, labels=legend)

    plt.show()


def plot_time_series(time, time_series, ylabel, title, figsize, color):
    plt.figure(figsize=figsize)
    plt.step(time, time_series, color=color)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.ylabel(ylabel)

    plt.show()


def plot_time_series_subplots(time, time_series_y1, time_series_y2, ylabel,
                              title, figsize, color, legend,
                              leg_loc="upper left"):
    plt.figure(figsize=figsize)

    plt.step(time, time_series_y1, label=legend[0], color=color[0])
    plt.step(time, time_series_y2, label=legend[1], color=color[1])

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.xlim(0, len(time_series_y1))

    plt.legend(loc=leg_loc)

    plt.show()

'''
Both pred and label are numpy nd matrix
'''
def accuracy(pred,label):
    count =0
    for i in range(len(label)):
        target = np.where(label[i]==1)
        arr_max = np.max(pred[i])
        idx = np.where(pred[i]==arr_max)[0]
        if target[:2] in idx:
            count = count+1

    acc = count/len(label)
    return acc
        



## initialize the feed forward network's weight
def Init_ForwardWgt(inputs, outputs, h, init =2):
    w_h = []
    w_o = []
    tmpp = np.random.normal(0, np.sqrt(3.0 / float(inputs)), [inputs, h[0]])
    if init == 1:
        cut = np.sqrt(3.0 / float(inputs)) * init
        tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
        tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(inputs))
        tmpp[tmpp > cut] = np.sqrt(3.0 / float(inputs))
    elif init == 2:
        if len(h) > 1:
            tmpp = np.random.normal(0, np.sqrt(6.0 / (float(inputs) + h[1])), [inputs, h[0]])
        else:
            tmpp = np.random.normal(0, np.sqrt(6.0 / (float(inputs) + outputs)), [inputs, h[0]])
    w_h.append(tmpp)
    for i in range(0, len(h) - 1):
        tmpp = np.random.normal(0, np.sqrt(3.0 / h[i]), [h[i], h[i + 1]])
        if init == 1:
            cut = np.sqrt(3.0 / float(h[i])) * init
            tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
            tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(h[i]))
            tmpp[tmpp > cut] = np.sqrt(3.0 / float(h[i]))
        elif init == 2:
            if (i + 2) < len(h):
                tmpp = np.random.normal(0, np.sqrt(6.0 / (h[i] + h[i + 2])), [h[i], h[i + 1]])
            else:
                tmpp = np.random.normal(0, np.sqrt(6.0 / (h[i] + outputs)), [h[i], h[i + 1]])
        w_h.append(tmpp)
    
    tmpp = np.random.normal(0, np.sqrt(3.0 / h[-1]), [h[-1], outputs])
    if init == 1:
        cut = np.sqrt(3.0 / float(h[-1])) * init
        tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
        tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(h[-1]))
        tmpp[tmpp > cut] = np.sqrt(3.0 / float(h[-1]))
    elif init == 2:
        tmpp = np.random.normal(0, np.sqrt(6.0 / h[-1]), [h[-1], outputs])
    w_o = tmpp
    return w_h, w_o


## initialize feedback (error) network's weight
def Init_FeedbackWgt(inputs, outputs, h, dfa =1):
    e_h = []
    e_o = []
    if dfa == 1:
        # tmpp = np.random.normal(0, np.sqrt(3.0 / float(h[0])),
        #                          size=[inputs, h[0]])
        tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(h[0])), high=np.sqrt(1.0 / float(h[0])), size=[inputs, h[0]])
        e_h.append(tmpp)
        for i in range(0, len(h) - 1):
            # tmpp = np.random.normal(0, np.sqrt(3.0 / h[i+1]), [h[i], h[i + 1]])

            tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(h[i+1])), high=np.sqrt(1.0 / float(h[i+1])),
                                     size=[h[i], h[i + 1]])
            e_h.append(tmpp)

        # tmpp = np.random.normal(0, np.sqrt(3.0 /float(outputs)), [h[-1], outputs])
        tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                 size=[h[-1], outputs])

        e_o = tmpp

    elif dfa == 2:
        # tmpp = np.random.normal(0, np.sqrt(1.0 / float(outputs)),
        #                          size=[inputs, outputs])
        tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                 size=[inputs, outputs])

        e_h.append(tmpp)
        for i in range(0, len(h) - 1):
            # tmpp = np.random.normal(0, np.sqrt(3.0 / float(outputs)), [h[i], outputs])

            tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                     size=[h[i], outputs])
            e_h.append(tmpp)

        # tmpp = np.random.normal(0, np.sqrt(3.0 /float(outputs)), [h[-1], outputs])
        tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                 size=[h[-1], outputs])

        # tmpp = np.random.rand(h[-1],outputs)/float(outputs.0)
        e_o = tmpp
        
    return e_h, e_o

### initialize thresholds for the forward and feedback network. They are initialized per layer based on the EMSTDP paper
def Init_Threshold(inputs, outputs, h, threshold_h, threshold_o, norm ,init =0, dfa =1 ):
    hiddenThr1 = threshold_h
    outputThr1 = threshold_o
    threshold_h = []
    # hThr1 = inputs*0.1
    hThr = inputs * np.sqrt(3.0 / float(inputs)) / (2.0)
    if init == 2:
        if len(h) > 1:
            hThr = inputs * np.sqrt(6.0 / (float(inputs) + h[1])) / 2.0
        else:
            hThr = inputs * np.sqrt(6.0 / (float(inputs) + outputs)) / 2.0

    threshold_h.append(hThr * hiddenThr1 / float(len(h) + 1))
    for i in range(len(h) - 1):
        if init != 2:
            threshold_h.append(h[i] * (np.sqrt(3.0 / h[i]) / 2.0) * hiddenThr1 / (len(h) - i - 0))
        elif init == 2:
            if (i + 2) < len(h):
                threshold_h.append(
                    h[i] * (np.sqrt(6.0 / (h[i] + h[i + 2])) / 2.0) * hiddenThr1 / (len(h) - i - 1))
            else:
                threshold_h.append(
                    h[i] * (np.sqrt(6.0 / (h[i] + outputs)) / 2.0) * hiddenThr1 / (len(h) - i - 1))

    if init != 2:
        threshold_o = h[-1] * (np.sqrt(3.0 / h[-1]) / 2.0) * outputThr1

    elif init == 2:
        threshold_o = h[-1] * (np.sqrt(6.0 / h[-1]) / 2.0) * outputThr1

    ethreshold_h = []
    ethreshold_o = []
    if init == 2:
        if len(h) > 1:
            hThr = h[1] * np.sqrt(6.0 / (float(inputs) + h[1])) / 2.0
        else:
            hThr = outputs * np.sqrt(6.0 / (float(inputs) + outputs)) / 2.0

        if norm == 0:
            ethreshold_h.append(hThr * hiddenThr1 / (len(h)))
        else:
            ethreshold_h.append(norm / (len(h)))

        for i in range(len(h) - 1):
            if (i + 2) < len(h):
                if norm == 0:
                    ethreshold_h.append(
                        h[i + 2] * (np.sqrt(6.0 / (h[i] + h[i + 2])) / 2.0) * hiddenThr1 / (i + 2))
                else:
                    ethreshold_h.append(norm / (i + 2))
            else:
                if norm == 0:
                    ethreshold_h.append(
                        outputs * (np.sqrt(6.0 / (h[i] + outputs)) / 2.0) * hiddenThr1 / (i + 2))
                else:
                    ethreshold_h.append(norm / (i + 2))

    if init != 2:
        ts = 1.0
        ehiddenThr1 = hiddenThr1
        eoutputThr1 = outputThr1
        for i in range(len(h)):
            if norm == 0:
                if i == len(h) - 1:

                    if dfa == 1:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))

                    elif dfa == 2:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                    else:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / h[-1]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                else:
                    if dfa == 1:
                        ethreshold_h.append(
                            h[i + 1] * (np.sqrt(3.0 / h[i + 1]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                    elif dfa == 2:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))

                    else:
                        ethreshold_h.append(
                            h[i + 1] * (np.sqrt(3.0 / h[i]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
            else:
                ethreshold_h.append(norm / (1))
    if init == 3:
        tss = 1.0
        ethreshold_h = np.divide(threshold_h, tss)
        ethreshold_o = np.divide(threshold_o, tss)

    return threshold_h, threshold_o, ethreshold_h, ethreshold_o

def plot_spikes_time_series(time, time_series, spikes, figsize, legend,
                            colors, title, num_steps):

    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps + 1, 25)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt.eventplot(positions=spikes,
                  lineoffsets=offsets,
                  linelength=0.9,
                  colors=colors)

    plt.title("Spike Arrival")
    plt.xlabel("Time steps")

    plt.xticks(num_x_ticks)
    plt.xlim(0, num_steps)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.yticks(ticks=offsets, labels=legend)
    plt.tight_layout(pad=3.0)

    plt.subplot(212)
    plt.step(time, time_series, color=colors)

    plt.title(title[0])
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.margins(x=0)

    plt.ylabel("Trace Value")

    plt.show()
