# Import Process level primitives.
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import matplotlib.pyplot as plt
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.resources import CPU
from lava.magma.core.model.model import AbstractProcessModel
# Import decorators.
from lava.magma.core.decorator import implements, tag, requires
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
# Import parent classes for ProcessModels for Hierarchical Processes.
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
import numpy as np
# Import execution protocol.
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
# Import monitoring Process.
from lava.proc.monitor.process import Monitor
'''
Neuron pass operation:
C.dv = b.s? C.u +a.v:0
dendrite a only pass the voltage
'''
class passDendrite(AbstractProcess):
    """Network of recurrently connected neurons.
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)



@implements(proc= passDendrite, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PypassDendriteModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (2**12 - self.du)*2**(-12)
        self.u[:] += a_in_data
        bias = self.bias_mant * (2 ** self.bias_exp)
        self.v[:] = self.v * (2**12 - self.dv)*2**(-12) + self.u + bias
        '''
        vth
        '''
        self.vth = self.vth*2**6
        s_out = self.v >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        self.s_out.send(self.v)
        


'''
Neuron pass operation:
C.dv = b.s? C.u +a.v:0
If input =0, d.v =0
else, dv = c.u+a.v
'''
class passSoma(AbstractProcess):
    """Network of recurrently connected neurons.
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)



@implements(proc=passSoma, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PypassSomaModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (2**12 - self.du)*2**(-12)
        self.u[:] += a_in_data
        '''
        Maybe need to modfifyed here
        a_in_data = 0 self.u =0
        a_in_data =1 self.u += slef.u+ b.v
        '''
        self.u[:] = a_in_data*self.u
        bias = self.bias_mant * (2 ** self.bias_exp)
        self.v[:] = self.v * (2**12 - self.dv)*2**(-12) + self.u + bias
        self.vth = self.vth*2**6
        s_out = self.v >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        self.s_out.send(s_out)
        


'''
Soma neuron
ds = a.s|b.s|c.s
Inhibitory connection to the dendrity[0] neuron
'''

class softSoma(AbstractProcess):
    """Network of recurrently connected neurons.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc= softSoma, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PysoftSomaModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.vth =self.vth*2**6
        s_out = a_in_data >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        self.s_out.send(s_out)

        

class softDendrite(AbstractProcess):
    """Network of recurrently connected neurons.
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc= softDendrite, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PysoftDendriteModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (2**12 - self.du)*2**(-12)
        self.u[:] += a_in_data
        bias = self.bias_mant * (2 ** self.bias_exp)
        #directly out_put the voltage, not the spike
        self.v[:] = self.v * (2**12 - self.dv)*2**(-12) + self.u + bias
        self.s_out.send(self.v)
        


'''
Soma neuron
ds = a.s|b.s|c.s
Inhibitory connection to the dendrity[0] neuron
'''

class andSoma(AbstractProcess):
    """Network of recurrently connected neurons.
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 20)

        # weight =10
        # vth = 20
        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape= shape, init=vth)


@implements(proc=andSoma, protocol=LoihiProtocol)
@requires(CPU)
@tag('emstdp')
class PyandSomaModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(np.ndarray, int)

    def run_spk(self):
        # the soma neurons recieves two ports input.
        # if the inputs is greater than the (spike+spike)*weight, then it fires
        a_in_data = self.a_in.recv()
        s_out = a_in_data >= self.vth
        self.s_out.send(s_out)

        

class andDendrite(AbstractProcess):
    """Network of recurrently connected neurons.
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc=andDendrite, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyandDendriteModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (2**12 - self.du)*2**(-12)
        self.u[:] += a_in_data
        bias = self.bias_mant * (2 ** self.bias_exp)
        self.vth =self.vth*2**6
        self.v[:] = self.v * (2**12 - self.dv)*2**(-12) + self.u + bias
        s_out = self.v >= self.vth
        self.s_out.send(s_out)