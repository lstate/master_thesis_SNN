#!/usr/bin/env python
# coding: utf-8

# # Training and Testing MNIST dataset

# In[1]:

import numpy as np
import math
import random
import cmath

# notes on comments: Q - Questions, A - Attention (to do, to pay attention to)


# In[2]:

# BASIC FUNCTIONS

# lif neuron with noise (Gaussian)
def LIF_step_noise(voltage, tau, current, dt, threshold, voltage_rest, resistance, variance_noise):
    if voltage < threshold:
        return (- voltage + current * resistance + voltage_rest) * dt / tau + voltage + np.random.normal(0, variance_noise, 1)
    else:
        return voltage_rest + np.random.normal(0, variance_noise, 1)
    
#grad decent (real numbers), quadratic error function, target function: product
def weight_update(weights, x, y, mu): 
    #mu learning rate
    return weights + 2 * mu * (y - np.dot(weights, x)) * x

#delay function (one step)
def delay_update(x, y): 
    return y - x

#grad decent (real numbers), quadratic error function, target function: sum
def delay_update_2(delays, x, y, mu): 
    #shape of delays: (neurons,)
    #mu learning rate
    return delays + 2 * mu * (y - (x + delays)) #ATTENTION: Y is a scalar, x and delays are vectors (but it works)!!!

#(complex) phase
def convert_phase(T, t):
    return math.pi * t / (T * 2) 

# time from (complex) phase
def convert_phase_back(T, phi):
    return phi * 2 * T / math.pi

# complex exponential for arrays
def convert_exp(array): 
    res = np.zeros(len(array), dtype=np.complex_)
    for a in range(len(array)):
        res[a] = cmath.exp(array[a])
    return res

# get phase, if negative convert to a positive number (3/4th quarter complex plane) -- not used (all results should be WITHIN 2 T_MAX)
def phase_2pi(complex_number):
    res = cmath.phase(complex_number)
    if res < 0: return (2 * math.pi + res)
    else: return res
    
# get also negative weights (3rd/4th quadrant complex space)
def real_phase(complex_number):
    res = cmath.phase(complex_number)
    if res < 0: 
        return (math.pi + res)
    else:
        return res
    
# get also negative weights (3rd/4th quadrant complex space)
def real_weights(complex_number):
    res = cmath.phase(complex_number)
    if res < 0: 
        return -abs(complex_number)
    else:
        return abs(complex_number)

# convert data to complex numbers (e.g. xdata)
def data_complex(neurons, X_delays, X_weights):
    # for one training/test example only (one value per neuron)
    complex_X = np.zeros((neurons, 1), dtype=np.complex_) # (neurons, 1) to make module 1 work
    for a in range(neurons):
        complex_X[a, 0] = cmath.exp(complex(0, 1) * convert_phase(t_max, X_delays[a])) * X_weights[a]
    return complex_X


# In[3]:

# PARAMETER SETTINGS LIF NEURON

dt = 0.01
tau = 5 
voltage_rest = 0
resistance = 1
variance_noise = 0.0 #variance Gaussian LIF neuron noise
learning_rate = 0.0001


# In[4]:

# MODULE 2 - TRAINING DELAYS AND WEIGHTS FOR THE GIVEN PATTERN

# module_2_3: mini batch grad descent for multiple postsyn neurons (first iterate over postsyn neurons, then over training data)
# module_2_3: adapted to data structure mnist

def module_2_3 (X_delays, X_weight, labels, learning_rate, iterations, neurons_pre, neurons_post):
    # array labels contains Y data (size_dataset x postsyn. neurons x 2)
    # index X data and Y data always minus one because otherwise first training example is not used !!
    d_evolution = np.zeros((neurons_post, neurons_pre, iterations + 1, 2))
    for a in range(iterations + 1):
        # first iteration: write everywhere zero (no change)
        for b in range(neurons_post):
            if a > 0:
                d_evolution[b, :, a, 1] = delay_update_2(d_evolution[b, :, a - 1, 1], X_delays[:, a - 1], labels[a - 1, b, 1], learning_rate)
                d_evolution[b, :, a, 0] = weight_update(d_evolution[b, :, a - 1, 0], X_weight[:, a - 1], labels[a - 1, b, 0], learning_rate)
    return d_evolution


# # READ TRAINING DATA

# In[5]:

# READ MNIST TRAINING DATA (X ONLY) AND DISPLAY

spikes = np.load('features_2/spikes_all_.txt.npy')
spike_times = np.load('features_2/spike_times_all_.txt.npy')
spike_weights = np.load('features_2/spike_weights_all_.txt.npy')

# PARAMETERS FROM TRAINING DATA

size_dataset = np.shape(spike_times)[0] #data set MNIST test digit, only binary spikes (no difference in weight!)
neurons_pre = np.shape(spike_times)[1] # presyn neurons
t_max = np.shape(spikes)[2] # t_max is the whole 'spike train' (left to right)


# In[6]:

# READ MNIST TRAINING DATA (Y ONLY) AND TRANSFORM THEM IN (WEIGHT, DELAY)

neurons_post = 10 # postsyn neurons
labels = np.loadtxt('features_2/indices_all.txt') # numbers between 0 and 9
labels_post = np.zeros((size_dataset, neurons_post, 2))

for a in range(size_dataset):
    labels_post[a, int(labels[a]), 0] = 1 # assign a weight of one to the postsyn neuron
    labels_post[a, :, 1] = t_max + t_max / 2
    labels_post[a, int(labels[a]), 1] = t_max + t_max / 2  # assign postsyn spike time 
    
print('read train data sucessfully')


# # READ TESTING DATA

# In[7]:

# READ MNIST TEST DATA 

feature_list = [20 *3, 60 *3, 100 * 3, 150 *3, 200 *3] #presyn neurons
points = len(feature_list)

spike_times_test = np.load('features_2/spike_times_all_test_.txt.npy') # (examples x neurons)
spike_weights_test = np.load('features_2/spike_weights_all_test_.txt.npy') # (examples x neurons)

#delays_test = np.transpose(spike_times_test)
#weights_test = np.transpose(spike_weights_test)

neurons_post = 10 # postsyn neurons
labels_test = np.loadtxt('features_2/indices_all_test.txt') # numbers between 0 and 9
size_dataset_test = np.shape(spike_times_test)[0]

print('read testing data successfully')


# # TRAIN AND TEST DATA

# In[9]:

# MAX MEMBRANE VOLTAGES (NO CUTTING AT THRESHOLD)

time_max = t_max * 3 # max time of the simulation
repetitions = 1 # number of repetitive calc. postsyn. potential

threshold = np.zeros((points, neurons_post))
accuracy = np.zeros((points))
confusion_matrix_ = np.zeros((points, neurons_post, neurons_post))

iterations = size_dataset

result_all_collect = []

for i in range(points):
    
    delays = np.transpose(spike_times[:, :feature_list[i]]) # (neurons x examples)
    weights = np.transpose(spike_weights[:, :feature_list[i]]) # (neurons x examples)
    result_all = module_2_3(delays, weights, labels_post, learning_rate, iterations, feature_list[i], neurons_post)
    
    test_range = 500 # 5 % of all testing data for spike statistics
    spike_label = np.zeros((neurons_post, size_dataset_test))
    max_voltage = np.zeros((test_range, neurons_post, 2)) 
    
    delays_test = np.transpose(spike_times_test[:, :feature_list[i]])
    weights_test = np.transpose(spike_weights_test[:, :feature_list[i]])

    for test in range(test_range):
        #membrane voltage postsyn
        voltage_steps = np.zeros((neurons_post, int(time_max/dt), repetitions))
        #input current (determined by spikes and respective delays) - postsyn.
        current_postsyn = np.zeros((neurons_post, int(time_max / dt))) 
        #input current (determined by spikes) - presyn.
        current_presyn = np.zeros((neurons_post, int(time_max / dt))) 
        #times of spikes postsyn, using result grad decent mini-batch
        #round spike times with a precision of 3 decimal numbers (corresponds to a resolution of 0.001)
        spike_times_post = np.zeros((neurons_post, feature_list[i])) 

        for a in range(neurons_post):
            spike_times_post[a, :] = delays_test[:, test] + result_all[a, :, iterations, 1]
            spike_times_post[a, :] = np.round(np.sort(spike_times_post[a, :]), decimals = 2)
            #input current post
            for b in range(feature_list[i]):
                if current_postsyn[a, int(spike_times_post[a, b] / dt)] == 0: 
                    current_postsyn[a, int(spike_times_post[a, b] / dt)] = result_all[a, b, iterations, 0] * weights_test[b, test] * tau / dt
                else: 
                    current_postsyn[a, int(spike_times_post[a, b] / dt)] = current_postsyn[a, int(spike_times_post[a, b] / dt)] + result_all[a, b, iterations, 0] * weights_test[b, test] * tau / dt

            #membrane potential
            #repetition: compute membrane potential several times as it is stochastic (noise)
            for c in range(repetitions):
                for d in range(int(time_max / dt)):
                    if d == 0:
                        voltage_steps[a, d, c] = voltage_rest
                    if d > 0:
                        voltage_steps[a, d, c] = LIF_step_noise(voltage_steps[a, d - 1, c], tau, current_postsyn[a, d], dt, 5, voltage_rest, resistance, variance_noise)
            # safe max voltage (one test example individual for every postsyn neuron)            
            max_voltage[test, a, 0] = np.max(voltage_steps[a, :, c])
            if labels_test[test] == a: max_voltage[test, a, 1] = 1
                
    print('extraction max volt successfully')

    # extract number of items per class (0...9)
    items = np.zeros(10) # stores number of items per class
    for a in range(test_range):
        for b in range(10):
            if labels_test[a] == b:
                items[b] = items[b] + 1
                
    # HISTOGRAM MAX MEMBRANE VOLTAGES + THRESHOLD
    threshold_list = []
    for h in range(neurons_post):
        list_class = []
        list_no_class = []
        # sort max voltages in lists according to their label 'belongs to class or not' (ONE POSTSYN NEURON: 0 / 1)
        for j in range(test_range):
            if max_voltage[j, h, 1] == 1:
                list_class.append(max_voltage[j, h, 0])
            else:
                list_no_class.append(max_voltage[j, h, 0])
        # write the same number of items each class in an array (ONE POSTSYN NEURON: 0 / 1)
        list_ = np.zeros((int(items[h] * 2), 2))
        list_[:int(items[h]), 0] = list_class
        list_[:int(items[h]), 1] = 1
        list_[int(items[h]):, 0] = list_no_class[:int(items[h])]
        sort_list = list_[np.argsort(list_[:, 0]), :]
        threshold_ = 0
        a = np.count_nonzero(list_[:, 1]) # count correct classification of 1s. Initially vth = 0, so always true
        b = 0 # count correct classification of 0s. Initially always wrong
        max_c = a + b  # number of right classifications
        for k, vl in enumerate(sort_list):
            if vl[1] == 0: #meaning that this input is not in the class
                b += 1   #this input would be correctly classified if vth = vl[0]
            else:
                a -= 1
            c = a + b
            if c > max_c:
                threshold_ = vl[0]
                max_c = c
        threshold_list.append(threshold_)  
        
    print('starting test all data')

# LIF NEURON GRAD DESCENT
    scaling = 1
    test_range_2 = size_dataset_test
    spike_label = np.zeros((neurons_post, size_dataset_test))
    max_voltage_2 = np.zeros((test_range_2, neurons_post))
    
    for test in range(test_range_2):

        #membrane voltage postsyn
        voltage_steps = np.zeros((neurons_post, int(time_max/dt), repetitions))
        # membrane voltage postsyn with no threshold (for WTA)
        voltage_steps_2 = np.zeros((neurons_post, int(time_max/dt), repetitions))
        #input current (determined by spikes and respective delays) - postsyn.
        current_postsyn = np.zeros((neurons_post, int(time_max / dt))) 
        #input current (determined by spikes) - presyn.
        current_presyn = np.zeros((neurons_post, int(time_max / dt))) 
        #times of spikes postsyn, using result grad decent mini-batch
        #round spike times with a precision of 3 decimal numbers (corresponds to a resolution of 0.001)
        spike_times_post = np.zeros((neurons_post, feature_list[i])) 

        for a in range(neurons_post):
            spike_times_post[a, :] = delays_test[:, test] + result_all[a, :, iterations, 1]
            spike_times_post[a, :] = np.round(np.sort(spike_times_post[a, :]), decimals = 2) 

            #input current post
            for b in range(feature_list[i]):
                if current_postsyn[a, int(spike_times_post[a, b] / dt)] == 0: 
                    current_postsyn[a, int(spike_times_post[a, b] / dt)] = result_all[a, b, iterations, 0] * weights_test[b, test] * scaling * tau / dt
                else: 
                    current_postsyn[a, int(spike_times_post[a, b] / dt)] = current_postsyn[a, int(spike_times_post[a, b] / dt)] + result_all[a, b, iterations, 0] * weights_test[b, test] * scaling * tau / dt

            #membrane potential
            #repetition: compute membrane potential several times as it is stochastic (noise), repetitions = 1 HERE
            for c in range(repetitions):
                for d in range(int(time_max / dt)):
                    if d == 0:
                        voltage_steps[a, d, c] = voltage_rest
                    if d > 0:
                        voltage_steps[a, d, c] = LIF_step_noise(voltage_steps[a, d - 1, c], tau, current_postsyn[a, d], dt, threshold_list[a], voltage_rest, resistance, variance_noise)
                        voltage_steps_2[a, d, c] = LIF_step_noise(voltage_steps[a, d - 1, c], tau, current_postsyn[a, d], dt, 5, voltage_rest, resistance, variance_noise)

                    if voltage_steps[a, d, c] > threshold_list[a]: # if there is a spike safe MAX VOLTAGE from trace with no threshold
                        max_voltage_2[test, a] = np.max(voltage_steps_2[a, :, c]) / threshold_list[a]

    # EVALUATE RESUTLS FROM LIF NEURON using 'winner-takes-all'
    confusion_matrix = np.zeros((neurons_post, neurons_post))
    count = 0
    for c in range(test_range_2):
        if max_voltage_2[c, int(labels_test[c])] > 0:
            maxneuron = np.argmax(max_voltage_2[c, :]) # WTA,filter position with max distance to threshold
            labelneuron = int(labels_test[c]) # correct neuron
            confusion_matrix[maxneuron, labelneuron] =  confusion_matrix[maxneuron, labelneuron] + 1
            if maxneuron == labelneuron:
                count = count + 1
    print('accuracy: %.2f' % (count / test_range_2))
    # normalize confusion matrix
    for d in range(10):
        norm = np.sum(confusion_matrix[:, d])
        confusion_matrix[:, d] = confusion_matrix[:, d] / norm
        
    #print(confusion_matrix)
        
    threshold[i, :] = threshold_list
    accuracy[i] = count / test_range_2
    confusion_matrix_[i, :, :] = confusion_matrix
    result_all_collect.append(result_all)
    
np.save('grad_descent_results_a.txt', accuracy)
np.save('grad_descent_results_c.txt', confusion_matrix_)
np.save('grad_descent_results_t.txt', threshold)


# In[ ]:



