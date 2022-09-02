# ann_models_library

This repository contains code that implements a python library to build artificial neural networks (ANN) from scratch. In particular, the library provides tools to find a decision boundary between two classes using a Multi-layer Percepton and to restored a denoised version of patterns using a Hopfield Network. 

### Getting startes (only for local usage)
#### Package Requirments

Installing following dependencies:
- python = 3.9
- intertools

#### Installation
Clone the repository

* https://github.com/ClaraEscorihuela/ann_models_library.git


### Models library:

This library implements 1-hidden layer MLP and a Hopfield Network from scratch. Available files in the library are as follows:

  1. Activations_functions: Contains a class called "default_activation" with activation functions to apply after each layer of the designed network. Possible activation functions:
  
    - Modified sigmoide
    - Derivative
                          
  2. Initialization_functions: Contains a class called "normal" that generates a normal gaussian dstribution with the desire mean and sigma. This class is used for the                                    initialization of the weights of the network. 
  
  3. Mlp_model: Contains a class called MLP to create and train a percepton with one hidden layer. This class includes:
                             
    - Arguments:
      * hidden_nodes: (int) number of nodes in the hidden layer
      * num_input: (int) dimension of one sample
      * num_out: (int) dimension output layer 
      * learning_rate (float)
      * weights (numpy array)
      * activation: (class) activation functions

     - Methods:
       * prepare_input function
       * initialize_weights
       * forwards_pass
       * backward_pass
       * weights_update
       * backpropagation algorithm
       * save_weights 
       * upload_weights
       * mse (mean square error)
       * accuracy 
       * get weights
       * predict
       * reset: It returns all arguments of the model to ther initial values
       * plot_decision_boundaries: Used in the demo explained above

  4. Hopfield_network: Contains a class called HopfieldNetwork to design a Hopfield Network (type of Recurrent Neural Network)
                              
    - Arguments:
      * W: (numpy array) weght matrix
      * positions_to_update: (numpy array) positions to update in the pattern 

     - Methods:
      * train: Calculate the weight matrix from input patterns
      * update: Update elements from a patern in syncronious or asyncroinious way
      * recall: Update patern until convergence
      * test_recall: Check if a pattern can converge to a fix value (base pattern)
                              
### Demonstration
The repository also contains a jupyter notebook (demo.py) with a short demonstration for the use of the models library. 
 
##### Exercise 1 : Multi Layer Perception 
The first exercise aims at designing an Percepton feedforward network with one hidden layer. The network can determine the decision boundary to differenciate two class using as input features the x-y position in a given space. This demo generates samples from two gaussian distributions with different mean and variance, and trains an MLP for classifying unseen samples from these distributions. This demo also shows how to apply a grid search for finding the best hyperparameters for training the model.
 
##### Exercise 2 : Hopfield Networks
The Hopfield Network is a type of recurrent neural network (RNN). One of its more interesting features is the associative memory; meaning that given a noisy pattern they can restore it to a trained or learned pattern. This demo aims at showing the use of the Hebbian rule as well as the synchronous and asynchronous method for the pattern update. 
 
                              
                              
                              
                              

