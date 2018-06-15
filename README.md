This is a program with some fundamental operations of Tensorflow. The data used here is mnist.

The fundamental operations include:
1. A fully-connected network with 2 layers.
2. Separated network definition file (mnist_train.py) and train process file (mnist_inference.py).
3. Learning rate with expontial decay.
4. Expontial moving average of variable.
5. Visualization by Tensorboard.
6. With/without regularization in loss function.
7. Average loss of each batch and each epoch in training process.
8. Early Stopping by either iterations or epoches.
9. Save the model and parameters.

The following is the content of scalars in Tensorboard. 'epoch_loss' is the average loss of each epoch, 'loss' includes 3 kind of losses of each iteration: cross_entropy loss, l2 loss, loss for mnist(cross_entropy loss + with/without l2 loss)

'layer1_scope' and 'layer2_scope' represent distribution of the weights and bias of the first and second layer, including min, max, mean and stddev. 'learning_rate' is the distribution of learning_rate in each epoch.

**cross_entropy loss + with l2 loss：**
![image](https://github.com/mna12478/Tensorflow_mnist/raw/master/tensorboard_cross_l2.png)

**cross_entropy loss + without l2 loss：**
![image](https://github.com/mna12478/Tensorflow_mnist/raw/master/tensorboard_cross.png)

##Reference
https://github.com/caicloud/tensorflow-tutorial
