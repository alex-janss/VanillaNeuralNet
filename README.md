# VanillaNeuralNet
This is a simple python 3 implementation of the vanilla neural network algorithm, used to classify the handwritten digits from the MNIST database. As my first python machine learning project, my goal was to understand the neural network and backpropagation algorithms in order to create a neural network in python from scratch, using only numpy.

### Prerequisites

After installing the latest version of Python 3 and a package manager such as pip, you will need the python packages numpy and matplotlib:

```
pip install numpy
pip install matplotlib
```

## Creating and testing the network

Run python, then create and train a new neural network (this may take a few minutes):
```
import NeuralNetwork.py
my_net = NeuralNet()
my_net.train()
```
Or, load the pre-trained network included in the repository:
```
import NeuralNetwork.py
my_net = pickle.load(open("example_net.pickle", "rb"))
```

Now, test the network's accuracy on the 10,000 test images. The example net should get an accuracy of about 95%:
```
my_net.test()
```
To manually input an image through the network and see the network's classification of that image, use the following code snippet:
```
# displaying the 45th image of the testing set and putting it through the network:
show_img(45, type="test")
my_net.classify(test_imgs[45],test_labels[45])
print("The network classifies this image as a ", np.argmax(my_net.neurons[-1]))
```
## Tuning the network hyper-parameters
Several parameters can be adjusted to optimize the network's performance.
The first is the network architecture: the number and size of each layer. By default the network has one hidden layer of size 100, for a total of 3 layers with 784, 100, and 10 neurons in each layer, respectively. To create a network with 2 hidden layers of size 150 and 40, run:
```
import NeuralNetwork.py
my_net = NeuralNet([784,150,40,10])
```
IMPORTANT NOTE: the first layer must have 784 neurons and the last layer must have 10 for the task of classifying MNIST digits.

In general, more and larger layers lead to better network accuracy but slower training. But having too many layers (more than 2 or 3) is not generally recommended, as it can reduce accuracy due to the vanishing/exploding gradient phenomenon.

The train() method accepts several parameters that can be tuned.
```
# the default parameters:
my_net.train(epochs=10, learn_rate=3.0, batch_size=10, init=True)
```
- `epochs` defines the number of times that the network is trained on the entire training set of 10,000 digits.
- `learn_rate` determines the "step size" that the network takes in the direction of the gradient vector. Too big a step size and the netowork will overshoot the local minimum, too small and it will learn slowly and possibly get stuck.
- `batch_size` determines the number of images in each stochastic batch. Larger batches guarantee a more accurate direction of descent, but will also slow the training.
- `init` determines whether the network will be randomly initialized (reset) before training. Set to `False` if the network has previously been initialized or trained.

For reference, the example network was created using `example_net = NeuralNet([784,300,10])` and trained with the following parameters: `example_net.train(50,3.0,10,True)`. This took about an hour on my machine.
