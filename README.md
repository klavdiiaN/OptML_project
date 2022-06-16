# Optimization for Machine Learning

This repository containes code and files for the project: "The impact of memory in optimization: how can a long-term memory improve the performance of Adam". The structure is as follows:
- Jupyter notebook [NosAdam_MNIST](NosAdam_MNIST.ipynb): search for the best parameters for NosAdam optimizer for MLP and CNN models using MNIST dataset,
- Jupyter notebook [Adam_MNIST_all_plots](Adam_MNIST_all_plots.ipynb): search for the best parameters for Adam optimizer for MLP and CNN models using MNIST dataset, the plots of convergence used in the report and the bootstrapping for both optimizers on MNIST,
- Jupter notebook [Adam_NosAdam_CNN_CIFAR10](Adam_NosAdam_CNN_CIFAR10.ipynb): search for the best parameters for both optimizers for CNN model using CIFAR10 dataset and corresponding plots of convergence,
- Folder [mnist_data](mnist_data): MNIST dataset of 50k train and 10k test samples,
- CIFAR10 dataset is too large for this repository but it can be downloaded [here](http://www.cs.toronto.edu/~kriz/cifar.html). Before running the code, the dataset should be unzipped and put to the same folder as the notebook with the code.
- The files [N_trials_adam_mlp](N_trials_adam_mlp.pth), [N_trials_adam_cnn](N_trials_adam_cnn.pth), [N_trials_nosadam_mlp](N_trials_nosadam_mlp.pth), [N_trials_nosadam_cnn](N_trials_nosadam_cnn.pth)
