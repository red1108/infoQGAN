import pennylane as qml
import torch
import numpy as np

class QGenerator:
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev, mode="RXYZ"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        self.mode = mode
        assert mode in ["RY", "RXYZ"], "mode should be 'RY' or 'RXYZ'"
        
    def init_circuit(self, generator_input):
        for i in range(self.n_qubits):
            qml.RY(generator_input[i]*np.pi/2, wires=i) # TODO: *a 해서 값 범위 맞추기
        
    def single_layer(self, params):
        if self.mode == "RY":
            for i in range(self.n_qubits):
                qml.RY(params[i][0], wires=i)
        elif self.mode == "RXYZ":
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

        return qml.state()

    def forward(self, generator_input):
        generator_output = [self.generator_circuit_qnode(single_in)
                            for single_in in generator_input]
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 2**output_qubits)
        return generator_output
    
    def parameters(self):
        return [self.params]
    
class QGAN2:
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev, entangling="CNOT"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        self.entangling = entangling
        assert entangling in ["CNOT", "CZ"], "entangling should be 'CNOT' or 'CZ'"
        
    def init_circuit(self, generator_input):
        for i in range(self.n_qubits):
            qml.RY(generator_input[i]*np.pi/2, wires=i) # TODO: *a 해서 값 범위 맞추기
        
    def single_layer(self, params, last=False):
        for i in range(self.n_qubits):
            qml.RY(params[i][0], wires=i)
        
        if not last:
            for i in range(self.n_qubits):
                if self.entangling == "CNOT":
                    qml.CNOT(wires=[i, (i+1)%self.n_qubits])
                elif self.entangling == "CZ":
                    qml.CZ(wires=[i, (i+1)%self.n_qubits])

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
    
    def parameters(self):
        return [self.params]
    
    def state_dict(self):
        return self.params

    
class QGAN2withSimpleForward(QGAN2):
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev, entangling="CNOT"):
        # Call parent constructor
        super().__init__(n_qubits, output_qubits, n_layers, params, dev, entangling)


    def circuit(self, generator_input):
        # output dimension: 2**output_qubits

        self.init_circuit(generator_input)

        for i in range(self.n_layers):
            self.single_layer(self.params[i], last=(i==self.n_layers-1))

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.output_qubits)]

    def forward(self, generator_input):
        generator_output = [self.generator_circuit_qnode(single_in) for single_in in generator_input]  # (BATCH_SIZE, output_qubits)
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, output_qubits)
        generator_output = (1 - generator_output) / 2 # (BATCH_SIZE, output_qubits) |1> 확률값을 구함
        return generator_output
    
class QGAN2withBitFlipNoise(QGAN2):
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev, noise_prob, entangling="CNOT"):
        # Call parent constructor
        super().__init__(n_qubits, output_qubits, n_layers, params, dev, entangling)
        self.noise_prob = noise_prob

    def init_circuit(self, generator_input):
        # For each qubit, apply RY followed by BitFlip noise
        for i in range(self.n_qubits):
            qml.RY(generator_input[i] * np.pi / 2, wires=i)
            qml.BitFlip(self.noise_prob, wires=i)

    def single_layer(self, params, last=False):
        # For each qubit, apply RY followed by BitFlip noise
        for i in range(self.n_qubits):
            qml.RY(params[i][0], wires=i)
            qml.BitFlip(self.noise_prob, wires=i)
            
        if not last:
            # Apply the entangling layer
            for i in range(self.n_qubits):
                if self.entangling == "CNOT":
                    qml.CNOT(wires=[i, (i+1) % self.n_qubits])
                elif self.entangling == "CZ":
                    qml.CZ(wires=[i, (i+1) % self.n_qubits])

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
    
    def parameters(self):
        return [self.params]
    
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
    
    def parameters(self):
        return [self.params]
    

class QGAN5:
    # RY RZ구조
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
            qml.RZ(params[i][1], wires=i)
        
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
    
    def parameters(self):
        return [self.params]
    

class QGAN6:
    # RY RZ구조 + Tracing out
    def __init__(self, n_qubits, output_qubits, n_layers, params, additional, dev):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.additional = additional
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        
    def init_circuit(self, generator_input):
        for i in range(self.n_qubits):
            qml.RY(generator_input[i]*np.pi/2, wires=i) # TODO: *a 해서 값 범위 맞추기
        for i in range(self.additional):
            for j in range(self.n_qubits - self.additional):
                qml.CNOT(wires=[self.n_qubits-i-1, j])
            
        
    def single_layer(self, params, last=False):
        for i in range(len(params)):
            qml.RY(params[i][0], wires=i)
            qml.RZ(params[i][1], wires=i)
        
        if not last:
            for i in range(len(params)):
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
    
    def parameters(self):
        return [self.params]
    

class QGAN2_ampl:
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
        qml.AmplitudeEmbedding(features=generator_input, wires=range(self.n_qubits))
        
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
    
    def parameters(self):
        return [self.params]
