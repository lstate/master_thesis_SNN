# master_thesis_SNN
code for master thesis: training delays in spiking neural networks

code files for the master thesis as jupyter notebook and python files, written in python 3

development of a new framework to train SNNs, based on two modules: 
module 1 (linear algebra approach) and module 2 (gradient descent)

## synthetic dataset: 
code for evaluating the new framework on a synthetic dataset and evaluating different types of noise (chapter 4 + 5)

- synthetic_dataset_1: noise on the spike times, weights and membrane potential, histograms of max current and max membrane pot
- synthetic_dataset_2: noise on the spike weights and influence on synaptic weights
- synthetic_dataset_basic: basic calculations and evalutaions
- complex_1: noise on spike time and influence on the modulus of the average complex weight

## mnist: 
code for evaluating framework on mnist, pre- and postprocessing (chapter 6)

folder 'non comp serv'
- state_space_weights_predicted_variables: state space weights, pred variables, histogram threshold extraction

folder 'comp serv'
- read_lables_mnist: extract labels from the original files and save them in a separate file
- data_mnist_comp (py file): feature selection (used for three different feature sets, executed on a computing server)
- results_both_2: evaluation plot 
- folder 'training and testing': code for training and testing (py files, with settings for training on feature set 2 and with tau = 5, executed on a computing server)


The full thesis is available on the preprint server of the MPI for Mathematics in the Sciences, Leipzig.
