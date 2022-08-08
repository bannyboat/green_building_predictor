# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 2021

@author: iuliia.glushakova@polyu.edu.hk

This script allows to call pre-trained models and by 
input data get predictions of cost or duration.
Note: this algorithm works for one project. If you want to predict several projects at once,
you'll have to modify it.
"""
import numpy as np
import joblib
from sklearn import svm
# import matplotlib.pyplot as plt


# Data normalization
class MinMaxNormalization:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        self.reference = None

    def normalize(self, array_to_normalize, reference_array=None):
        if type(array_to_normalize) ==  list:
            array_to_normalize = [float(i) for i in array_to_normalize]
        array_to_normalize = np.atleast_2d(array_to_normalize)  # If data was sent as a list it will transform it into an array

        # If there is a reference array use it as reference
        if not np.any(reference_array):
            self.reference = array_to_normalize
        else:
            self.reference = reference_array
        
        m, n = array_to_normalize.shape
        # if n > m:
        #     array_to_normalize = array_to_normalize.reshape(n, m)
        row, col = array_to_normalize.shape      
        
        self.minimum = np.zeros(col)
        self.maximum = np.zeros(col)
        for i in range(0, col):
            self.minimum[i] = min(self.reference[:, i])
            self.maximum[i] = max(self.reference[:, i])

        normalized_array = np.zeros((row, col))
        for r in range(0, row):
            for c in range(0, col):
                normalized_array[r, c] = (array_to_normalize[r, c] - self.minimum[c]) / (self.maximum[c] - self.minimum[c])

        return normalized_array

    def reverse(self, array_to_reverse, original_reference=None):
        array_to_reverse = np.atleast_2d(array_to_reverse)
        m, n = array_to_reverse.shape
        if n > m:
            array_to_reverse = array_to_reverse.reshape(n, m)
        row, col = array_to_reverse.shape

        # If there is a reference array use it as reference
        if np.any(original_reference):
            self.reference = original_reference
        if not np.any(self.reference):
            print("You need to normalize the array first or set the 'original_reference' parameter.")

        # Calculating min and max values
        self.minimum = np.zeros(col)
        self.maximum = np.zeros(col)
        for i in range(0, col):
            self.minimum[i] = min(self.reference[:, i])
            self.maximum[i] = max(self.reference[:, i])

        denormalized_array = np.zeros((row, col))  # New, denormalized array
        for r in range(0, row):
            for c in range(0, col):
                denormalized_array[r, c] = (array_to_reverse[r, c] * (self.maximum[c] - self.minimum[c])) + self.minimum[c]
        return denormalized_array


# Accuracy for regression problem NN
class Accuracy:
    def __init__(self):
        self.precision = None
        self.sum_comparisons = 0

    def calculate(self, prediction, target):
        comparisons = self.compare(prediction, target)
        accuracy = np.mean(comparisons)
        self.sum_comparisons += np.sum(comparisons)
        self.samples_count += len(comparisons)
        return accuracy

    def accuracy_summary(self):
        accuracy = self.sum_comparisons / self.samples_count
        return accuracy

    def reset(self):
        self.sum_coparisons = 0
        self.samples_count = 0

    def initialize(self, target, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(target) / 250

    def compare(self, prediction, target):
        return np.absolute(prediction - target) < self.precision


### Layers
class Hidden_Layer:
    def __init__(self, num_inputs, num_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        # Initializing weights and biases
        self.weights = 0.01 * np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

        # Setting regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Pass forward
    def forward(self, inputs, is_training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Pass backward
    def backward(self, dvalues):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Get layer's weights and biases
    def get_weights_and_biases(self):
        return self.weights, self.biases

    # Set weights and biases
    def set_weights_and_biases(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Input_Layer:
    def forward(self, inputs, is_training):
        self.output = inputs


# Activation functions
class Activation:
    def forward(self, inputs, is_training):
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues

    def predictions(self, outputs):
        return outputs


class ReLU(Activation):
    def forward(self, inputs, is_training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient for negative input values
        self.dinputs[self.inputs <=0] = 0

    def predictions(self, outputs):
        return outputs


class SoftMax(Activation):
    def forward(self, inputs, is_training):
        self.dinputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise array
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Linear(Activation):
    def forward(self, inputs, is_training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs
        

class Loss:
    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
            return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, target, *, include_regularization=False):
        sample_losses = self.forward(output, target)
        data_loss = np.mean(sample_losses)
        self.sum_losses += np.sum(sample_losses)
        self.counted_samples += len(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def losses_summary(self, *, include_regularization=False):
        data_loss = self.sum_losses / self.counted_samples
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def reset(self):
        self.sum_losses = 0
        self.counted_samples = 0


class MeanSquareError(Loss):
    def forward(self, prediction, target):
        sample_losses = np.mean((target - prediction)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, target):
        num_samples = len(dvalues)
        num_outputs = len(dvalues[0])   # Number of outputs in every sample
        self.dinputs = -2 * (target - dvalues) / num_outputs
        self.dinputs = self.dinputs / num_samples

class MeanAbsoluteError(Loss):
    def forward(self, prediction, target):
        sample_losses = np.mean(np.abs(target - prediction), axis=1)
        return sample_losses

    def backward(self, dvalues, target):
        num_samples = len(dvalues)
        num_outputs = len(dvalues[0])
        self.dinputs = np.sign(target - dvalues) / num_outputs
        self.dinputs = self.dinputs / num_samples


# Optimiser
class Optimizer:
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        pass

    def post_update_params(self):
        self.iterations += 1


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Since self.iteration is 0 at the start, add 1
        updated_weight_momentums = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        updated_bias_momentums = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        update_weight_cache = layer.weight_cache / (1 - self.beta_2 **(self.iterations + 1))
        update_bias_cache = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * updated_weight_momentums / \
                         (np.sqrt(update_weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * updated_bias_momentums / (np.sqrt(update_bias_cache) + self.epsilon)


### NN model
class NN:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set_functions(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def build(self):
        self.input_layer = Input_Layer()
        num_layers = len(self.layers)   # Number of layers
        self.trainable_layers = []  # Layers which weights and biases will change
        # Add previous (prev) and following (next) layers to every layer of the model
        for l in range(num_layers):
            # Input layer
            if l == 0:
                self.layers[l].prev = self.input_layer
                self.layers[l].next = self.layers[l + 1]
            elif l < (num_layers - 1):
                self.layers[l].prev = self.layers[l - 1]
                self.layers[l].next = self.layers[l + 1]
            # The next layer for the last hidden layer is loss and then output layer
            else:
                self.layers[l].prev = self.layers[l - 1]
                self.layers[l].next = self.loss
                self.output_layer_activation = self.layers[l]
            # If a layer has weights, it's trainable layer
            if hasattr(self.layers[l], "weights"):
                self.trainable_layers.append(self.layers[l])
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

    def train(self, train_input, train_target, *, epochs=1, batch_size=None, print_frequency=1, validation_input=None, validation_output=None):
        self.accuracy.initialize(train_target)
        train_steps = 1
        if batch_size is not None:
            train_steps = len(train_input) // batch_size
            if train_steps * batch_size < len(train_input):
                train_steps += 1
        # Training
        for epoch in range(1, epochs+1):
            #if not epoch % print_frequency or epoch == 1:
            print(f"Epoch: {epoch}")
            self.loss.reset()
            self.accuracy.reset()

            for step in range(train_steps):
                if batch_size is None:
                    input_batch = train_input
                    target_batch = train_target
                else:
                    input_batch = train_input[step * batch_size:(step + 1) * batch_size]
                    target_batch = train_target[step * batch_size:(step + 1) * batch_size]
                output = self.forward(input_batch, is_training=True)
                data_loss, regularization_loss = self.loss.calculate(output, target_batch, include_regularization=True)
                loss = data_loss + regularization_loss
                prediction = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(prediction, target_batch)
                self.backward(output, target_batch)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Printing training results
                if print_frequency != 0:
                    if not step % print_frequency or step == (train_steps - 1):
                        print(f"step: {step}, " +
                              f"accuracy: {accuracy:.3f}, " +
                              f"loss: {loss:.3f}, (" +
                              f"data loss: {data_loss:.3f}, " +
                              f"regularization loss: {regularization_loss:.3f}), " +
                              f"learning rate: {self.optimizer.current_learning_rate:.8f};")
            epoch_data_loss, epoch_regularization_loss = self.loss.losses_summary(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.accuracy_summary()
            print(f"training, " +
                  f"accuracy: {epoch_accuracy:.3f}, " +
                  f"loss: {epoch_loss:.3f}, (" +
                  f"data loss: {epoch_data_loss:.3f}, " +
                  f"regularization loss: {epoch_regularization_loss:.3f}), " +
                  f"learning rate: {self.optimizer.current_learning_rate}.")

            if validation_input is not None:
                self.evaluate(valid_input=validation_input, valid_target=validation_output, batch_size=batch_size)

    def evaluate(self, valid_input, valid_target, *, batch_size=None):
        valid_steps = 1
        if batch_size is not None:
            valid_steps = len(valid_input) // batch_size
            if valid_steps * batch_size < len(valid_input):
                valid_steps += 1
        self.loss.reset()
        self.accuracy.reset()
        for step in range(valid_steps):
            if batch_size is None:
                input_batch = valid_input
                target_batch = valid_target
            else:
                input_batch = valid_input[step * batch_size: (step + 1) * batch_size]
                target_batch = valid_target[step * batch_size: (step + 1) * batch_size]

            output = self.forward(input_batch, is_training=False)
            self.loss.calculate(output, target_batch)
            prediction = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(prediction, target_batch)

        valid_loss = self.loss.losses_summary()
        valid_accuracy = self.accuracy.accuracy_summary()

    def predict(self, input, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(input) // batch_size
            if prediction_steps * batch_size < len(input):
                prediction_steps += 1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                input_batch = input
            else:
                input_batch = input[step * batch_size:(step + 1) * batch_size]
            batch_output = self.forward(input_batch, is_training=False)
            output.append(batch_output)
        return  np.vstack(output)   # Stack arrays in sequence vertically (row wise).

    def forward(self, input, is_training:bool):
        self.input_layer.forward(input, is_training)
        for layer in self.layers:
            layer.forward(layer.prev.output, is_training)
        return  layer.output

    def backward(self, output, target):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, target)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, target)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def get_weights_and_biases(self):
        wb_array = []
        for layer in self.trainable_layers:
            wb_array.append(layer.get_weights_and_biases())
        return wb_array

    def set_weights_and_biases(self, wb_array):
        for wb_set, layer in zip(wb_array, self.trainable_layers):
            layer.set_weights_and_biases(*wb_set) 
        
    # Calculating number of neurons on hidden layers
    def calculate_neurons(self, num_layers, training_input, is_num_neurons_based_on_features=False):
        self.num_layers = num_layers    # Set the number of layers for NN
        num_samples, num_features = training_input.shape
        relative_number = num_features * is_num_neurons_based_on_features + num_samples * (not is_num_neurons_based_on_features)
        
        num_neurons = np.zeros(num_layers, dtype=int)
        for l in range(num_layers):
            if l == 0:
                num_neurons[l] = relative_number * 2 + 1
            elif l == (num_layers-1):
                num_neurons[l] = 1
            else:
                num_neurons[l] = num_neurons[l - 1] // 2    # Getting the closest integer
        return num_neurons


def calculate_days_of_weather(start_month, start_year, duration, weather_condition):

    weather_table = np.loadtxt(get_weather_table(weather_condition))
    
    # Check if data us array or a value
    if type(start_month) == int:
        is_array = False
        k = 1    # Number of projects
    if type(start_month) == np.array:
        is_array = True
        k = len(start_month)    # Number of projects
    n = len(weather_table)  # Number of years in the weather table
    days_count = np.zeros(k)   # Number of days of days count
    
    # Find the start year in the weather table
    for l in range(0, n):                
        for i in range(0, k):
            if is_array:
                check_year = start_year[i]
                pd = duration[i]
                sm = start_month[i]
                sy = start_year[i]
            else:
                check_year = start_year
                pd = duration
                sm = start_month
                sy = start_year

            if weather_table[l, 0] == check_year:
                
                n_years = int(round(pd))       
                n_months = int(round(pd * 12))  # Number of months spent on the project, rounded to the closest integer
                count_months = n_months
                # Here and forth printing functions are commented, because it is here for debugging purposes
                # print("Duration of the project in months: ",n_months)
            
                for d in range(l, (l + n_years + 1)):   # Add one because the months not always start or end at January
                    if d < n:                        
                        # print("l = ", d, "; l+n_years = ", (l+n_years))
                        if weather_table[d, 0] == sy:
                            # print("Starting from ", weather_table[d,0], " year.")
                            # print("Starting month: ", sm)
                            for j in range(sm, 13): 
                                # print("Current month:", j)
                                if count_months > 0:
                                    if is_array:
                                        days_count[i] = days_count[i] + weather_table[d,j]
                                    else:
                                        days_count = days_count + weather_table[d,j]
                                    count_months -=1
                                    # print("Months left: ", count_months)
                                    
                        if weather_table[d,0] != sy:
                            # print("Current year: ", weather_table[d,0])
                            for j in range (1, 13):  
                                if count_months > 0:
                                    # print("Current month: ", j)
                                    if is_array:
                                        days_count[i] = days_count[i] + weather_table[d,j]
                                    else:
                                        days_count = days_count + weather_table[d,j]
                                    count_months -=1
                                    # print("Months left: ", count_months)
                    else:
                        print("The planned duration is exceeding ", n, " the available weather information. \n\n")
                        break
    return days_count

def get_weather_table(weather_condition):
    switch = {
        "cold": "Weather/Cold_days.txt",  # Data on cold days from HK Observatory
        "hot": "Weather/Hot_days.txt",  # Data on hot days
        "rain": "Weather/Rain_days.txt"  # Data on rainy days    
    }
    return switch.get(weather_condition,"When calling 'calculate_days_of_weather()' input weather condition parameter as \"cold\", \"hot\" or \"rain\".")

# This is method that calls pre-trained SVR.
def call_svr(input_data, problem="FC"):
    
    with open(f"data/svr_config/{problem}/svr_model.pkl", "rb") as svr_model:
        # Call load method to deserialze
        svr = joblib.load(svr_model)
    prediction = svr.predict(input_data).reshape(-1, 1)
    return prediction


# This method calls pre-trained DNN.
# The results of DNN prediction are sent to SVR as inputs. You don't need to modify it, unless you want ot change the name for "problem" variable.
def call_nn(input_data, problem="FC"):  # problem is a srting that tell NN what we are trying to predict: FC - cost, AD - duration.
    
    # Loading DNN config according to the "problem"
    with open(f"data/nn_config/{problem}/nn_model.pkl", "rb") as nn_model:
            # Call load method to deserialze
            nn = joblib.load(nn_model)
    dnn_output = nn.predict(input_data).reshape(-1, 1)   
    
    return dnn_output


# This is the method that combines all AI techniques and gives the final prediction
# Feed input data and problem (as either "FC" of "AD") and you'll get the result
def predict(input_data, problem="FC"):
    # First data is normalised using references
    # Loading reference data
    reference_data = np.load("data/Reference_data.npy")  # Contains all the GBPs' data. With project number at the first column, AC and AD at the last 2 columns
    columns = np.atleast_2d(reference_data).shape[1]  # Calculate the number of columns in the reference data 
    
    # Since teh reference data contains all data, we use positioning to get the right values out of the reference
    if problem == "FC":
        output_pos = 0  # Position of the FC in the output array, counting from the last
    elif problem == "AD":
        output_pos = 1  # Position of the AD in the output array, counting from the last
    else:
        print(f"The \"{problem}\" problem is out of range of this framework. Please try \"FC\" or \"AD\".")
        return
    output_reference = reference_data[:, columns - 2 + output_pos].reshape(-1, 1)    # Reference for output, exludes columns that are used in input
    input_reference = np.delete(reference_data, [0, columns - 2, columns - 1], axis=1)  # Reference for input features, excluding project number and targets.
    mmn_input = MinMaxNormalization()  # Normalisation object for input
    mmn_output = MinMaxNormalization()  # Normalisation object for output

    # Normalising
    norm_input = mmn_input.normalize(input_data, input_reference)  # Normalised input
    dnn_output = call_nn(norm_input, problem=problem)  # Calling DNN on normalised input data
    framework_result = call_svr(dnn_output, problem)
    prediction = mmn_output.reverse(framework_result, output_reference)
    return prediction