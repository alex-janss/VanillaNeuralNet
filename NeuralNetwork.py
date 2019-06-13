import pickle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=300)
image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
# Edit data path to your own directory if needed
data_path = ""
# Saves train and test data to pickle file
# Makes loading faster for future runs
try:
    train_data = pickle.load(open("train_data.pickle", "rb"))
    test_data = pickle.load(open("test_data.pickle", "rb"))
except (OSError, IOError) as e:
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
    pickle.dump(train_data, open("train_data.pickle", "wb"))
    pickle.dump(test_data, open("test_data.pickle", "wb"))

fac = .99 / 255
# separate images and labels, map each pixel value into range [.01,1]
train_imgs = np.asfarray(train_data[:, 1:]) * fac + .01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + .01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(no_of_different_labels)
# transform labels into one hot representation
train_labels = (lr == train_labels).astype(np.float)
test_labels = (lr == test_labels).astype(np.float)


# Sigmoid activation function. Takes input in range (-inf,inf) and outputs in range (0,1)
# This is a variation of the traditional y=1/(1+exp(x)) function.
# I found this version to work slightly better
def sigmoid(input_vector):
    return np.arctan(input_vector) / np.pi + .5


# derivative of the sigmoid function
def deriv_sigmoid(input_vector):
    return 1. / (np.pi * (1 + input_vector ** 2))


# shuffles a pair of vectors together,
# i.e. both vectors are shuffled exactly the same way
def shuffle_together(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

class NeuralNet:

    # Constructor
    # Initalizes all neurons, weights and biases to zero.
    def __init__(self, input_params=[image_pixels,100,no_of_different_labels], batch_size=1):
        # "params" defines the network architecture.
        # by default, the net has 3 layers with 784 neurons
        # in the first layer, 100 in the 2nd, and 10 in the 3rd.
        # A network can be created with any number and size of layers,
        # but for the task of classifying MNIST digits the first layer
        # must have 784 neurons and the last layer must have 10.
        self.params = input_params
        #
        self.neurons = []
        self.weighted_inputs = []
        self.weights = []
        self.biases = []
        for i in range(len(self.params) - 1):
            self.weights.append(np.zeros((self.params[i + 1], self.params[i])))
            self.biases.append(np.zeros((self.params[i + 1])))
        for i in range(len(self.params)):
            self.neurons.append(np.zeros((self.params[i], batch_size)))
            self.weighted_inputs.append(np.zeros((self.params[i], batch_size)))
            self.errors = np.zeros((self.params[-1], batch_size))

    # randomly initializes all weights and biases normally with variance 1
    def random_init(self):
        for i in range(len(self.params) - 1):
            self.weights[i] = np.random.randn(self.params[i + 1], self.params[i])
            self.biases[i] = np.random.randn(self.params[i + 1])

    # calculates the cost (sum of squared errors) or average cost of a batch
    def get_cost(self):
        if 1 in self.errors.shape:
            return la.norm(self.errors)
        else:
            return sum(la.norm(self.errors, axis=0))/self.errors.shape[1]

    # add weights and biases of two networks
    def adjust_net(self, gradient):
        for i in range(len(self.params) - 1):
            self.weights[i] += gradient.weights[i]
            self.biases[i] += gradient.biases[i]

    # scale all weights and biases by some factor
    def scale_net(self, factor):
        for i in range(len(self.params) - 1):
            self.weights[i] *= factor
            self.biases[i] *= factor
        return self

    # Classifies an input image or batch of images by propagating it forward through the net.
    # Multiple image vectors can be classified in one pass using matrix methods.
    # The network's "choice" of an image's classification can be
    # determined directly by np.argmax(self.neurons[-1]).
    def classify(self, input_imgs, input_labels):
        if input_imgs.ndim == 1:
            input_imgs.shape = (1, self.params[0])
            input_labels.shape = (1, self.params[-1])
        self.neurons[0] = input_imgs.T
        for i in range(len(self.params) - 1):
            self.weighted_inputs[i + 1] = (self.weights[i] @ self.neurons[i]) + np.outer(self.biases[i], np.ones((self.neurons[0].shape[1])))
            self.neurons[i+1] = sigmoid(self.weighted_inputs[i + 1])
        self.errors = self.neurons[-1] - input_labels.T


    # Uses the backpropagation algorithm to compute the negative gradient w.r.t the cost
    # stored in neural net object "gradient". This object is not meant
    # to be used as a neural net, it is simply a convenient way to store
    # the adjustments needed for each of the weights and biases.
    # This implementation is capable of back-propagating multiple error vectors
    # in one pass using matrix methods.
    def backprop(self):
        batch_size = self.errors.shape[1]
        gradient = NeuralNet(input_params=self.params, batch_size=batch_size)
        for i in range(-1, -len(self.params), -1):
            # neurons error (derivative of the cost w.r.t each neuron activation)
            if i != -1:
                gradient.neurons[i] = (self.weights[i + 1].T @ gradient.neurons[i + 1]) * deriv_sigmoid(self.weighted_inputs[i])
            else:
                gradient.neurons[-1] = 2 * self.errors * deriv_sigmoid(self.weighted_inputs[-1])
            # bias error (derivative of the cost w.r.t each bias)
            gradient.biases[i] = np.sum(gradient.neurons[i], 1) / batch_size
            # weight error (derivative of the cost w.r.t. each weight)
            temp = np.zeros((batch_size, self.params[i], self.params[i-1]))
            for j in range(batch_size):
                temp[j] = np.outer(gradient.neurons[i][:, j], self.neurons[i - 1][:, j])
            gradient.weights[i] = np.sum(temp, 0) / batch_size
        # negative gradient:
        gradient.scale_net(-1)
        return gradient


    # Train the network using stochastic gradient descent,
    # with stochastic batches of size "batch_size",
    # taking steps proportional to the learn_rate.
    # A learn_rate of 3 worked well for the default network
    # This will run through the entire training set
    # "epochs" number of times
    def train(self, epochs=10, learn_rate=3.0, batch_size=10, init=True):
        # randomly initialize the network
        if init:
            self.random_init()
        print("Displaying avg cost each epoch:")
        for k in range(epochs):
            shuffle_together(train_imgs, train_labels)
            avg_cost = 0.0
            # divide the training set into (10000/batch_size) batches
            for i in range(10000 // batch_size):
                # classify a batch of images
                self.classify(train_imgs[i * batch_size:i * batch_size + batch_size], train_labels[i * batch_size:i * batch_size + batch_size])
                # calculate the average cost for the batch
                avg_cost += self.get_cost()
                # calculate the gradient vector via backprop(),
                # then scale the gradient by learn_rate and add to the net
                self.adjust_net(self.backprop().scale_net(learn_rate))
            # average over the number of batches
            avg_cost /= (10000/batch_size)
            print('%.5f'%avg_cost, "\tepoch", k+1)

    # Tests the network on all test_imgs.
    # prints the confusion matrix, the average least-squares cost, and the accuracy.
    def test(self):
        avg_cost = 0.0
        cm = np.zeros((10,10), dtype= int)
        for i in range(10000):
            self.classify(test_imgs[i], test_labels[i])
            avg_cost += self.get_cost()
            cm[np.argmax(test_labels[i]),np.argmax(self.neurons[-1])] += 1
        avg_cost /= 10000
        accuracy = cm.trace()/10000
        print("Confusion Matrix:")
        print(cm)
        print("average cost: ", avg_cost)
        print("accuracy: ", accuracy*100, "%")

# --------- end Neural Net class ----------


# displays an image in grayscale.
def show_img(index, type="train"):
    if type == "train":
        img = train_imgs[index].reshape((28, 28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        print(train_labels[index])
    elif type == "test":
        img = test_imgs[index].reshape((28, 28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        print(test_labels[index])
    else:
        print("parameter error")




