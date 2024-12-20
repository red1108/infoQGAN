#%%
import numpy as np
from modules.utils import generate_orthogonal_states



n = 4 # dimension of the state
m = 3 # number of states
states = generate_orthogonal_states(n, m)
print(states.shape)
# check orthogonality for all states (complex inner product)
print(states @ states.T.conj())

#%%
import pennylane as qml

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(state=None):
    qml.StatePrep(
        state, wires=range(2))
    return qml.expval(qml.Z(0)), qml.state()


# %%
