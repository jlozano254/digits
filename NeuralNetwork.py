import math
import random
import time
import numpy as np

class NeuralNetwork:

    digits = [
        [
            1,1,1,1,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,1,1,1,1
        ],
        [
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0
        ],
        [
            1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,1
        ],
        [
            1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1
        ],
        [
            1,0,0,0,1,
            1,0,0,0,1,
            1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,0,1
        ],
        [
            1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1
        ],
        [
            1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1
        ],
        [
            1,1,1,1,1,
            0,0,0,1,0,
            0,0,1,0,0,
            0,1,0,0,0,
            1,0,0,0,0
        ],
        [
            1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1
        ],
        [
            1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,0,1
        ]
    ]

    base_output = [
        [1,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,1]
    ]

    show_operations = False

    def __init__(self, seed = 5, alpha = 0.1, min_error_percentage = 0.0005, input_size = 25, output_size = 10, hidden_num = 5):
        self.seed = seed
        self.alpha = alpha
        self.min_error_percentage = min_error_percentage
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_num = hidden_num
    
    def withSeed(self, seed):
        self.seed = seed
        return self

    def withAlpha(self, alpha):
        self.alpha = alpha
        return self

    def withMinErrorPercentage(self, min_error_percentage):
        self.min_error_percentage = min_error_percentage
        return self

    def verbose(self, show_operations):
        self.show_operations = show_operations
        return self

    def withHiddenLabels(self, hidden_num):
        self.hidden_num = hidden_num
        return self

    def randomize(self):
        random.seed(self.seed)
        neural_network = [
            [
                [random.randint(-1, 0) for _ in range(self.input_size + 1)] for _ in range(self.hidden_num)
            ],
            [
                [random.randint(-1, 0) for _ in range(self.hidden_num + 1)] for _ in range(self.output_size)
            ]
        ]
        return neural_network
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def product(self, v, w):
        return sum([a * b for a, b in zip(v, w)])

    def neuron_output(self, weights, inputs):
        return self.sigmoid(self.product(weights, inputs))

    def ffnn(self, neural_network, inputs):
        outputs = []
        for label in neural_network:
            inputs = inputs + [1]
            output = [self.neuron_output(neuron, inputs) for neuron in label]
            outputs.append(output)
            inputs = output
        return outputs

    def back_propagation(self, digit, inputs, target):
        hidden_output, output = self.ffnn(digit, inputs)
        new_output = []
        new_hidden = []
        
        error = sum((output - target) * (output - target) for output, target in zip(output, target)) * 0.5
        delta_output = [output * (1 - output) * (output - target) for output, target in zip(output, target)]
        
        for i, output_neuron in enumerate(digit[-1]):
            for j, hidden_output_current in enumerate(hidden_output + [1]):
                output_neuron[j] -= delta_output[i] * hidden_output_current * self.alpha
            new_output.append(output_neuron)
            if (self.show_operations):
                print("Neuron weights: ", i, output_neuron)
        
        hidden_delta = [hidden_output_current * (1 - hidden_output_current) * self.product(delta_output, [n[i] for n in digit[-1]]) for i, hidden_output_current in enumerate(hidden_output)]
        
        for i, hidden_neuron in enumerate(digit[0]):
            for j, input_ in enumerate(inputs + [1]):
                hidden_neuron[j] -= hidden_delta[i] * input_ * self.alpha
            new_hidden.append(hidden_neuron)
            if (self.show_operations):
                print("Hidden neuron weights: ", i, hidden_neuron)

        return new_hidden, new_output, error 
    
    def randomTraining(self):
        print("Starting training...")
        start = time.time()
        output = self.randomize()
        sq_error = 1
        iterations = 1

        while sq_error > self.min_error_percentage:
            sq_error = 0
            for i in range(len(self.digits)):
                hidden, output, error = self.back_propagation(output, self.digits[i], self.base_output[i])
                output = [hidden, output]
                sq_error += error
            sq_error = sq_error / len(self.digits)
            if (self.show_operations):
                print("Iterations: ", iterations, ", error percentage: ", sq_error)
            iterations += 1
        
        self.output_data = output
        end = time.time()
        elapsed = end - start
        print("Trained finished in ", elapsed, " seconds")

    def guessWith(self, output):
        index = 0
        closest_dif = abs(output[0] - 1)
        for i, value in enumerate(output):
            current_dif = abs(value - 1)
            if (current_dif < closest_dif):
                closest_dif = current_dif
                index = i
        return index

    def test(self, input_):
        result = self.ffnn(self.output_data, input_)[-1]
        print("Output: ", result)
        print("Your number probably is: ", self.guessWith(result))
