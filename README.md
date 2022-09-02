# ann_models_library

This repository aims at showing how to use a library with artificial neural networks models build from scratch, as well as how to aply an MLP to find a decision boundary between two classes and how to apply a Hopfield Network. 

### Available files in the models library:

  1. activations_functions: Contains a class called "default_activation" with activation functions to apply after each layer of the designed network. Possible activation functions:
  
    - Modified sigmoide
    - Derivative
                          
  2. initialization_functions: Contains a class called "normal" that generates a normal gaussian dstribution with the desire mean and sigma. This class is used for the                                    initialization of the weights of the network. 
  
  3. mlp_model: Contains a class called MLP to create and train a percepton with one hidden layer. This class includes:
                             
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

  4. hopfield_network: Contains a class called HopfieldNetwork to design a Hopfield Network (type of Recurrent Neural Network)
                              
    - Arguments:
      * W: (numpy array) weght matrix
      * positions_to_update: (numpy array) positions to update in the pattern 

     - Methods:
      * train: Calculate the weight matrix from input patterns
      * update: Update elements from a patern in syncronious or asyncroinious way
      * recall: Update patern until convergence
      * test_recall: Check if a pattern can converge to a fix value (base pattern)
                              
### Demonstration
The repository contains a jupyter notebook with a short demonstration for the use of the modes library. 
 
##### Exercise 1 : Multi Layer Perception 
The first exercise aims at designing an Percepton feedforward network with one hidden layer. The network must be capable to determine the decision boundary and to differenciate two class samples from their position in a given space. This demo generates samples from two gaussian distributions with different mean and variance, and trains an MLP for classifyin unseen samples from these distributions. This demo also shows how to apply a grid search for finding the best hyperparameters for the model.
 
##### Exercise 2 : Hopfield Networks
The Hopfield Network is one king of recurrent neural networks (RNN). RNNs are characterize by the entrance of their output signals as input signals. One of its more interesting feautres is the associative memory, given a noisy pattern they can restore it to a trained or learned pattern. This demo aims at showing the training and the recall method to learn and restorn the trained patterns. 
 
                              
                              
                              
                              

