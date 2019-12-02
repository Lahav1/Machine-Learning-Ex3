import numpy as np
import sys

PIXELS_NUM = 28 * 28
HIDDEN_LAYER_SIZE = 100
LABELS_NUM = 10
EPOCHS_NUM = 3000
ETA = 5.3


def rearrange_y(train_y):
    """
    Rearanges the y_train vector (size: 1 x num of inputs) to a matrix (size: num of inputs x num of labels),
    which every index in the matrix represents the value of the output and marked by 1,
    and the rest are zeros.

    Parameters
    ----------
    train_y : array of strings.

    Returns
    -------
    new_train_y : train_y as a matrix.
    """
    new_train_y = np.zeros((len(train_y), LABELS_NUM))
    for i in range(len(train_y)):
        index = train_y[i]
        new_train_y[i][index] = 1
    return new_train_y


def relu(x):
    """
    Normalizes the matrix x by ReLU function.

    Parameters
    ----------
    x : matrix with values in range [0, 255].

    Returns
    -------
    matrix with normalized values.
    """
    # normalize the values between [0,1]
    x /= np.max(x)
    # return the absolute value of x if positive, and 0 if negative.
    return abs(x) * (x > 0)


def relu_derivative(x):
    """
    Apply the ReLU derivative function to a matrix.

    Parameters
    ----------
    x : matrix.

    Returns
    -------
    New matrix which each value is derived.
    """
    return x > 0


def softmax(x):
    """
    Normalizes the matrix x by softmax function.

    Parameters
    ----------
    x : matrix with values in range [0, 255].

    Returns
    -------
    matrix with normalized values.
    """
    x -= np.max(x)
    exp = np.exp(x).T
    s = np.sum(np.exp(x), axis=1)
    return exp / s


class NeuralNetwork:
    def __init__(self, train_x, train_y):
        """
                    Data Members:
        x:                      train input.
        correct_y:              train output.
        inputs_num:             number of train data.
        eta:                    learning rate.
        epochs:                 number of iterations.
        hidden_layer:           layer no. 1.
        output_layer:           layer no. 2.
        weights1, weights2:     weight matrices.
        bias1, bias2:           bias vectors.
        gradients:              gradients dictionary.
        """
        # normalize inputs in range [0,1]
        self.x = train_x.T / np.max(train_x)
        self.inputs_num = len(train_x)
        # pad the train_y matrix with zeros.
        self.correct_y = rearrange_y(train_y).T
        self.eta = ETA
        self.epochs = EPOCHS_NUM
        # initialize values:
        # initialize both layers.
        self.hidden_layer_size = HIDDEN_LAYER_SIZE
        self.hidden_layer = np.zeros((HIDDEN_LAYER_SIZE, self.inputs_num))
        self.output_layer = np.zeros((LABELS_NUM, self.inputs_num))
        # randomize values for w1, w2, b1, b2.
        self.weights1 = np.random.randn(HIDDEN_LAYER_SIZE, PIXELS_NUM)
        self.weights2 = np.random.rand(LABELS_NUM, HIDDEN_LAYER_SIZE)
        self.bias1 = np.random.rand(HIDDEN_LAYER_SIZE, 1)
        self.bias2 = np.random.rand(LABELS_NUM, 1)
        self.gradients = {}

    def f_prop(self, inputs):
        """
        Forward propagation stage.

        Parameters
        ----------
        inputs : train_x inputs (if training) or test_x inputs (if predicting).

        Returns
        -------
        Updated hidden layer and output layer.
        """
        # create local variables of parameters to work with.
        local_weights1 = self.weights1
        local_weights2 = self.weights2
        local_bias1 = self.bias1
        local_bias2 = self.bias2
        # apply relu activation function on (w1*x + b1)
        h_l = relu(np.dot(local_weights1, inputs) + local_bias1)
        # apply relu activation function on (w2*hidden_layer + b2)
        o_l = softmax((np.dot(local_weights2, h_l) + local_bias2).T)
        return h_l, o_l

    def b_prop(self):
        """
        Backward propagation stage.
        """
        # create local variables of parameters to work with.
        local_weights2 = self.weights2
        local_x = self.x
        local_hidden_layer = self.hidden_layer
        local_output_layer = self.output_layer
        # compute new values.
        den = (1 / self.inputs_num)
        dz2 = local_output_layer - self.correct_y
        dw2 = den * np.dot(dz2, local_hidden_layer.T)
        db2 = den * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.multiply(np.dot(local_weights2.T, dz2), relu_derivative(local_hidden_layer))
        dw1 = den * np.dot(dz1, local_x.T)
        db1 = den * np.sum(dz1, axis=1, keepdims=True)
        # update the gradients.
        self.gradients["weights1"] = dw1
        self.gradients["weights2"] = dw2
        self.gradients["bias1"] = db1
        self.gradients["bias2"] = db2

    def update(self):
        """
        Update the value members using eta, weight and bias.
        """
        # create local variables of parameters to work with.
        local_weights1 = self.weights1
        local_weights2 = self.weights2
        local_bias1 = self.bias1
        local_bias2 = self.bias2
        # update the values using eta value and gradients.
        self.weights1 = local_weights1 - self.eta * self.gradients["weights1"]
        self.weights2 = local_weights2 - self.eta * self.gradients["weights2"]
        self.bias1 = local_bias1 - self.eta * self.gradients["bias1"]
        self.bias2 = local_bias2 - self.eta * self.gradients["bias2"]

    def loss(self):
        """
        Calculate the loss using NNL loss function.
        """
        local_output_layer = self.output_layer
        loss = - np.sum(np.multiply(np.log(local_output_layer), self.correct_y) / self.inputs_num)
        return loss

    def train(self):
        """
        Train the neural network.
        """
        for i in range(self.epochs):
            # call forward & backward propagation and then update the values.
            self.hidden_layer, self.output_layer = self.f_prop(self.x)
            self.b_prop()
            self.update()
            if i % 200 == 0:
                # from iteration 1500 and on, shrink eta's value.
                if i > 1600:
                    self.eta = self.eta * 0.9
            # each 100 iterations, calculate the loss and accuracy to follow the changes.
            if i % 100 == 0:
                # follow changes in loss and print them.
                loss = self.loss()
                print("Iteration: %i, Loss: %f" % (i, loss))

    def predict(self, test_x):
        """
        Use the trained neural network to predict test_x's labels.
        """
        # get the prediction matrix by applying the forward propagation function on test_x.
        h_l, o_l = self.f_prop(test_x)
        # create a predictions vector by taking the arg-max of each output.
        return np.argmax(o_l, axis=0)


def main():
    np.random.seed(1)
    # get file paths from program arguments.
    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    # load data to matrices.
    train_x = np.loadtxt(train_x_path, dtype=np.int, delimiter=' ')
    train_y = np.loadtxt(train_y_path, dtype=np.int, delimiter='\n')
    test_x = np.loadtxt(test_x_path, dtype=np.int, delimiter=' ')
    # create a neural network and train it.
    nn = NeuralNetwork(train_x, train_y)
    nn.train()
    # after training, get the predictions for the test set.
    predictions = nn.predict(test_x.T)
    predictions = predictions.T
    predictions.astype(int)
    np.savetxt("D:\\ex3_files\\test_y", predictions, delimiter='\n', fmt='%d')


main()
