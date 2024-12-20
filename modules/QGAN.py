import pennylane as qml
import torch
import numpy as np

class QGenerator:
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev, give_me_states=False):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        self.give_me_states = give_me_states
        
    def init_circuit(self, generator_input):
        for i in range(self.n_qubits):
            qml.RY(generator_input[i]*np.pi/2, wires=i) # TODO: *a 해서 값 범위 맞추기
        
    def single_layer(self, params):
        for i in range(self.n_qubits):
            qml.RX(params[i][0], wires=i)
            qml.RY(params[i][1], wires=i)
            qml.RZ(params[i][2], wires=i)
        
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i+1)%self.n_qubits])

    def circuit(self, generator_input):
        # output dimension: 2**output_qubits

        self.init_circuit(generator_input)

        for param in self.params:
            self.single_layer(param)

        if self.give_me_states:
            return qml.probs(wires=range(self.output_qubits)), qml.state()
        return qml.probs(wires=range(self.output_qubits)) # |00>, |01>, |10>, |11> 이런식으로 모든 basis들의 확률값을 반환

    def forward(self, generator_input):
        if self.give_me_states:
            all_probs = []
            all_states = []
            for single_in in generator_input:
                # QNode에서 (probs, state)를 반환
                probs, state = self.generator_circuit_qnode(single_in)
                all_probs.append(probs)
                all_states.append(state)
            all_probs = torch.stack(all_probs)  # (BATCH_SIZE, 2**output_qubits)
            all_states = torch.stack(all_states)  # (BATCH_SIZE, 2**n_qubits)
            return all_probs, all_states

        else:
            # give_me_states=False이면, probs만 반환
            generator_output = [self.generator_circuit_qnode(single_in)
                                for single_in in generator_input]
            generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 2**output_qubits)
            return generator_output
    
class QGAN2:
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        
    def init_circuit(self, generator_input):
        for i in range(self.n_qubits):
            qml.RY(generator_input[i]*np.pi/2, wires=i) # TODO: *a 해서 값 범위 맞추기
        
    def single_layer(self, params, last=False):
        for i in range(self.n_qubits):
            qml.RY(params[i][0], wires=i)
        
        if not last:
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i+1)%self.n_qubits])

    def circuit(self, generator_input):
        # output dimension: 2**output_qubits

        self.init_circuit(generator_input)

        for i in range(self.n_layers):
            self.single_layer(self.params[i], last=(i==self.n_layers-1))

        return qml.probs(wires=range(self.output_qubits)) # |00>, |01>, |10>, |11> 이런식으로 모든 basis들의 확률값을 반환

    def forward(self, generator_input):
        generator_output = [self.generator_circuit_qnode(single_in) for single_in in generator_input]  # (BATCH_SIZE, 2**output_qubits)
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 2**output_qubits)
        return generator_output

class QGAN3:
    # 여러 깊이로 seed를 임베딩할수 있게 함.
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        
    def init_circuit(self, generator_seed):
        for i in range(self.n_qubits):
            qml.RY(generator_seed[i]*np.pi/2, wires=i) # TODO: *a 해서 값 범위 맞추기
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i+1)%self.n_qubits])
        
    def single_layer(self, params, cnot=False):
        for i in range(self.n_qubits):
            qml.RY(params[i][0], wires=i)
        
        if cnot:
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i+1)%self.n_qubits])

    def circuit(self, generator_seed):
        # generator_seed: (n_qubits, dim)
        # output dimension: 2**output_qubits

        seed_depth = generator_seed.shape[1]
        for i in range(seed_depth):
            self.init_circuit(generator_seed[:,i])

        for i in range(self.n_layers):
            self.single_layer(self.params[i], cnot=(i<self.n_layers-1))

        return qml.probs(wires=range(self.output_qubits)) # |00>, |01>, |10>, |11> 이런식으로 모든 basis들의 확률값을 반환

    def forward(self, generator_seed):
        generator_output = [self.generator_circuit_qnode(single_in) for single_in in generator_seed]  # (BATCH_SIZE, 2**output_qubits)
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 2**output_qubits)
        return generator_output
    
class QGAN4:
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        
    def init_circuit(self, generator_seed):
        qml.AmplitudeEmbedding(features=generator_seed, wires=range(self.n_qubits), normalize=True, pad_with=0.0)
        
    def single_layer(self, params, cnot=False):
        for i in range(self.n_qubits):
            qml.RY(params[i][0], wires=i)
        
        if cnot:
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i+1)%self.n_qubits])

    def circuit(self, generator_seed):
        # generator_seed: (2**n_qubits)
        # output dimension: 2**output_qubits
        self.init_circuit(generator_seed)
        for i in range(self.n_layers):
            self.single_layer(self.params[i], cnot=(i<self.n_layers-1))

        return qml.probs(wires=range(self.output_qubits)) # |00>, |01>, |10>, |11> 이런식으로 모든 basis들의 확률값을 반환

    def forward(self, generator_seed):
        generator_output = [self.generator_circuit_qnode(single_in) for single_in in generator_seed]  # (BATCH_SIZE, 2**output_qubits)
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 2**output_qubits)
        return generator_output
    


class QDiscriminator():
    def __init__(self, n_qubits, n_layers, params, dev):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        
    def init_circuit(self, initial_state):
        qml.StatePrep(initial_state, pad_with=0., validate_norm=True, wires=range(self.n_qubits))
        
    def single_layer(self, params):
        for i in range(self.n_qubits):
            qml.RX(params[i][0], wires=i)
            qml.RY(params[i][1], wires=i)
            qml.RZ(params[i][2], wires=i)
        
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i+1)%self.n_qubits])

    def circuit(self, generator_input):
        # output dimension: 2**output_qubits

        self.init_circuit(generator_input)

        for param in self.params:
            self.single_layer(param)

        return qml.probs(wires=0) # 첫 큐빗이 [0일 확률, 1일 확률]

    def forward(self, initial_states):
        generator_output = [self.generator_circuit_qnode(initial_state)[0] for initial_state in initial_states]  # (BATCH_SIZE, 1)
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 1)
        return generator_output