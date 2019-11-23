# Digits
Digit prediction using neural network with back propagation algorithm &amp; sigmoid function

## Usage

```python
from NeuralNetwork import *
network = NeuralNetwork()
```

## Config
```python
network.verbose(False) # Deactivate print messages
network.withSeed(5)    # Seed for random numbers
network.withAlpha(0.1) # Set alpha const
network.withMinErrorPercentage(0.0005) # Set percentage error tolerance
network.withHiddenLabels(5) # Set hidden labels number
```

## Training
```python
network.randomTraining() # The network gets trained by random seed
```

## Some tests
```python
network.test(
    [
        0,0,1,1,0,
        0,1,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0
    ]
) # This should predict that the number is 0

network.test(
    [
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,0,1,0
    ]
) # This should predict that the number is 1

network.test(
   [
       1,1,1,1,1,
       0,0,0,0,1,
       1,1,1,1,0,
       1,0,0,0,0,
       1,1,1,1,1 
  ]
) # This should predict that the number is 2

network.test(
   [
       1,1,1,1,1,
       0,0,0,0,1,
       0,1,1,1,1,
       0,0,0,0,1,
       0,1,1,1,1 
  ]
) # This should predict that the number is 3

network.test(
   [
       1,1,0,0,1,
       1,1,0,0,1,
       1,1,1,1,1,
       0,0,0,0,1,
       0,0,0,0,1 
  ]
) # This should predict that the number is 4

network.test(
   [
        0,1,1,1,1,
        0,1,0,0,0,
        0,1,1,1,0,
        0,0,0,0,1,
        1,1,1,1,1
  ]
) # This should predict that the number is 5

network.test(
    [
        1,1,1,1,0,
        1,0,0,0,0,
        1,1,1,1,0,
        1,0,0,1,0,
        1,1,1,1,0
    ]
) # This should predict that the number is 6

network.test(
    [
        1,1,1,1,1,
        0,0,1,0,0,
        0,1,0,0,0,
        0,1,0,0,0,
        1,0,0,0,0
    ]
) # This should predict that the number is 7

network.test(
    [
        0,1,1,1,0,
        1,0,0,0,1,
        0,1,1,1,0,
        1,0,0,0,1,
        0,1,1,1,0
    ]
) # This should predict that the number is 8

network.test(
    [
        1,1,1,1,1,
        1,1,0,1,1,
        1,1,1,1,1,
        0,0,0,1,1,
        0,0,0,1,1
    ]
) # This should predict that the number is 9
```
