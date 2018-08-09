# Somewhat Keras Tutorial

## Viewing tensors as vectors with their geometry

In general, elementary geometric operations such as affine transformations, rotations,
scaling, and so on can be expressed as tensor operations. For instance, a rotation of a
2D vector by an angle theta can be achieved via a dot product with a 2 × 2 matrix
R = [u, v] , where u and v are both vectors of the plane: u = [cos(theta),
sin(theta)] and v = [-sin(theta), cos(theta)] .


# What is a derivative?

Consider a continuous, smooth function f(x) = y , mapping a real number x to a new real number y . Because the function is continuous, a small change in x can only result in a small change in y —that’s the intuition behind continuity. Let’s say we increase x by a small factor epsilon_x : this results in a small epsilon_y change to y :
f(x + epsilon_x) = y + epsilon_y

In addition, because the function is smooth (its curve doesn’t have any abrupt angles), when epsilon_x is small enough, around a certain point p , it’s possible to approximate f as a linear function of slope a , so that epsilon_y becomes a * epsilon_x :
f(x + epsilon_x) = y + a * epsilon_x
Obviously, this linear approximation is valid only when x is close enough to p . The slope a is called the derivative of f in p . If a is negative, it means a small change of x around p will result in a decrease of f(x) (as shown in figure 2.10); and if a is positive, a small change in x will result in an increase of f(x) . Further, the absolute value of a (the magnitude of the derivative) tells we how quickly this increase or decrease will happen.


# What is gradient?

A gradient is the derivative of a tensor operation. It’s the generalization of the concept of derivatives to functions of multidimensional inputs: that is, to functions that take tensors as inputs.
Consider an input vector x , a matrix W , a target y , and a loss function loss . we can use W to compute a target candidate y_pred , and compute the loss, or mismatch, between the target candidate y_pred and the target y.


## A nice example for NN

In 3D , the following mental image may prove useful. Imagine two sheets of colored paper: one red and one blue. Put one on top of the other. Now crumple them together into a small ball. That crumpled paper ball is our input data, and each sheet of paper is a class of data in a classification problem. What a neural network (or any other machine-learning model) is meant to do is figure out a transformation of the paper ball that would uncrumple it, so as to make the two classes cleanly separable again. With deep learning, this would be implemented as a series of simple transformations of the 3D space, such as those we could apply on the paper ball with our fingers, one movement at a time.
Uncrumpling paper balls is what machine learning is about: finding neat representations for complex, highly folded data manifolds. At this point, we should have a pretty good intuition as to why deep learning excels at this: it takes the approach of incrementally decomposing a complicated geometric transformation into a long
chain of elementary ones, which is pretty much the strategy a human would follow to uncrumple a paper ball. Each layer in a deep network applies a transformation that disentangles the data a little—and a deep stack of layers makes tractable an extremely complicated disentanglement process.


# The usual training loop process

Training, is basically the learning that machine learning is all about. This happens within what’s called a training loop, which works as follows. Repeat these steps in a loop, as long as necessary:

1. Draw a batch of training samples x and corresponding targets y .
2. Run the network on x (a step called the forward pass) to obtain predictions y_pred .
3. Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y .
4. Update all weights of the network in a way that slightly reduces the loss on this batch.


## General process

1. Draw a batch of training samples x and corresponding targets y .
2. Run the network on x to obtain predictions y_pred .
3. Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y .
4. Compute the gradient of the loss with regard to the network’s parameters (a backward pass).
5. Move the parameters a little in the opposite direction from the gradient—for example W -= step * gradient —thus reducing the loss on the batch a bit.

## Symbolic differentiation

Nowadays, and for years to come, people will implement networks in modern frameworks that are capable of symbolic differentiation, such as TensorFlow. This means that, given a chain of operations with a known derivative, they can compute a gradient function for the chain (by applying the chain rule) that maps network parameter values to gradient values. When we have access to such a function, the backward pass is reduced to a call to this gradient function. Thanks to symbolic differentiation, we’ll never have to implement the Backpropagation algorithm by hand.


## Anatomy of a Neural Net in Keras

As we saw in the previous chapters, training a neural network revolves around the fol-
lowing objects:
1. Layers, which are combined into a network (or model)
2. The input data and corresponding targets
3. The loss function, which defines the feedback signal used for learning.
4. The optimizer, which determines how learning proceeds

# Layers as lego blocks

we can think of layers as the LEGO bricks of deep learning, a metaphor that is made explicit by frameworks like Keras. Building deep-learning models in Keras is done by clipping together compatible layers to form useful data-transformation pipelines. The notion of layer compatibility here refers specifically to the fact that every layer will only accept input tensors of a certain shape and will return output tensors of a certain shape.


# Model: Network of Layers

A deep-learning model is a directed, acyclic graph of layers. The most common
instance is a linear stack of layers, mapping a single input to a single output. But as we move forward, we’ll be exposed to a much broader variety of network topologies. Some common ones include the following:
1. Two-branch networks
2. Multihead networks
3. Inception blocks
The topology of a network defines a hypothesis space. we may remember that in chapter 1, we defined machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, we constrain our space of possibilities (hypothesis space) to a specific series of tensor operations, mapping input data to output data.

**Picking the right network architecture is more an art than a science; and although there are some best practices and principles we can rely on, only practice can help we become a proper neural-network architect.**

## Loss and Optimizers

A neural network that has multiple outputs may have multiple loss functions (one per output). But the gradient-descent process must be based on a single scalar loss value; so, for multiloss networks, all losses are combined (via averaging) into a single scalar quantity. Choosing the right objective function for the right problem is extremely important: our network will take any shortcut it can, to minimize the loss; so if the objective doesn’t fully correlate with success for the task at hand, our network will end up doing things we may not have wanted.


## What are activation functions and why are they necessary?

Without an activation function like relu (also called a non-linearity), the Dense layer would consist of two linear operations—a dot product and an addition:

`output = dot(W, input) + b`

So the layer could only learn linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space. Such a hypothesis space is too restricted and wouldn’t benefit from multiple layers of representations, because a deep stack of linear layers would still implement a linear operation: adding more layers wouldn’t extend the hypothesis space.

In order to get access to a much richer hypothesis space that would benefit from deep representations, we need a non-linearity, or activation function. relu is the most popular activation function in deep learning, but there are many other candidates, which all come with similarly strange names: prelu, elu, and so on.


## Few words on Keras

Keras has the following key features:
1. It allows the same code to run seamlessly on CPU or GPU .
2. It has a user-friendly API that makes it easy to quickly prototype deep-learning models.
3. It has built-in support for convolutional networks (for computer vision), recurrent networks (for sequence processing), and any combination of both.
4. It supports arbitrary network architectures: multi-input or multi-output models, layer sharing, model sharing, and so on. This means Keras is appropriate for building essentially any deep-learning model, from a generative adversarial net- work to a neural Turing machine.


Keras is a model-level library, providing high-level building blocks for developing deep-learning models. It doesn’t handle low-level operations such as tensor manipulation and differentiation. Instead, it relies on a specialized, well-optimized tensor library to do so, serving as the backend engine of Keras. Rather than choosing a single tensor library and tying the implementation of Keras to that library, Keras handles the problem in a modular way; thus several different backend engines can be plugged seamlessly into Keras.


# Workflow of Keras

1. Define our training data: input tensors and target tensors.
2. Define a network of layers (or model ) that maps our inputs to our targets.
3. Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor.
4. Iterate on our training data by calling the fit() method of our model.


# [Making our First Net]()

## Step 1: Defining Model Type
There are two ways to define a model: using the **Sequential class** (only for linear stacks of layers, which is the most common network architecture by far) or the **functional API** (for directed acyclic graphs of layers, which lets we build completely arbitrary architectures)

### Example

Defining same architecture for the _sequential class_ and _functional api_.
> In my personal opinion using the _functional api_ gives much more flexibility and is a lot more visibily traceable.

**Sequential class**
```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
```

**Functional API**
```python
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)

output_tensor = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=input_tensor, outputs=output_tensor)
```

With the functional API , we’re manipulating the data tensors that the model processes and applying layers to this tensor as if they were functions.

Once our model architecture is defined, it doesn’t matter whether we used a _Sequential model_ or the _functional API_. All of the following steps are the same.

## Step 2: Loss and Optimizer

The learning process is configured in the compilation step, where we specify the optimizer and loss function(s) that the model should use, as well as the metrics we want to monitor during training. Here’s an example with a single loss function, which is by far the most common case:

```python
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])
```

## Step 3: Coming together

Finally, the learning process consists of passing Numpy arrays of input data (and the corresponding target data) to the model via the _fit()_ method, similar to what we would do in Scikit-Learn and several other machine-learning libraries:

```python
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
```

### There are a few key architecture decisions to be made about such a stack of Dense layers

1. How many layers to use?
2. How many hidden units to choose for each layer?
3. Which activation to choose?

## [Movie reviews example]()

Binary classification problem.

### Preparing Text data

we can’t feed lists of integers into a neural network. we have to turn our lists into tensors. There are two ways to do that:
1. **Intelligent way** Pad our lists so that they all have the same length, turn them into an integer tensor of shape (samples, word_indices) , and then use as the first layer in our network a layer capable of handling such integer tensors (the Embedding layer).
2. **Bleh!** One-hot encode our lists to turn them into vectors of 0s and 1s. This would mean, for instance, turning the sequence [3, 5] into a 10,000-dimensional vector that would be all 0s except for indices 3 and 5, which would be 1s. Then we could use as the first layer in our network a Dense layer, capable of handling floating-point vector data


### Take aways

1. we usually need to do quite a bit of preprocessing on our raw data in order to be able to feed it as tensors into a neural network. Sequences of words can
be encoded as binary vectors, but there are other encoding options, too.

2. Stacks of Dense layers with relu activations can solve a wide range of problems
(including sentiment classification), and we’ll likely use them frequently.

3. In a binary classification problem (two output classes), popular network should
end with a Dense layer with one unit and a sigmoid activation: the output of
our network should be a scalar between 0 and 1, encoding a probability.

4. With such a scalar sigmoid output on a binary classification problem, the loss
function we should use is binary_crossentropy .

5. The rmsprop optimizer is generally a good enough choice, whatever our prob-
lem. That’s one less thing for we to worry about.

6. As they get better on their training data, neural networks eventually start over-
fitting and end up obtaining increasingly worse results on data they’ve never
seen before. Be sure to always monitor performance on data that is outside of
the training set.


## [News Classification]()

Multiclass classification problem.

In the previous example, we saw how to do binary classification, let's see what happens for multiclass outputs.

We'll use Reuters dataset, It’s a simple, widely used toy dataset for text classification. There are 46 different topics; some topics are more represented than others, but each topic has at least 10 examples in the training set.

###  Another way of encoding categorical data

Another way to encode the labels would be to cast them as, an integer tensor, like this:

```python
y_train = np.array(train_labels)
y_test = np.array(test_labels)
```
The only thing this approach would change is the choice of the loss function. The loss
function used [here](), categorical_crossentropy , expects the labels to follow
a categorical encoding. With integer labels, we should use `sparse_categorical_crossentropy` :

```python
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
```
This new loss function is still mathematically the same as categorical_crossentropy; it just has a different interface.


###  Why have large intermediate layers??

In the [new_classification]() project we used large intermediate layers, to avoid a architectural information bottleneck. To see what happens when we introduce an information bottleneck by having intermediate layers that are significantly less than 46-dimensional, try this:

```python
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))
```
The network would now be peaking at **~71% validation accuracy, an 8% absolute drop**. This drop is mostly due to the fact that we’re trying to compress a lot of information (enough information to recover the separation hyperplanes of 46 classes) into an intermediate space that is too low-dimensional. The model is able to cram most of the necessary information into these eight-dimensional representations, but not all of it.


### Things that came out

1. The network ends with a Dense layer of size 46. This means for each input sample, the network will output a 46-dimensional vector. Each entry in this vector (each dimension) will encode a different output class.

2. The last layer uses a softmax activation. we saw this pattern in the MNIST example. It means the network will output a probability distribution over the 46 different output classes for every input sample, the network will produce a 46 dimensional output vector, where output[i] is the probability that the sample belongs to class i. The 46 scores will sum to 1.

3. If we’re trying to classify data points among N classes, our network should end with a Dense layer of size N .

4. In a single-label, multiclass classification problem, our network should end with a softmax activation so that it will output a probability distribution over the N output classes.

5. Categorical crossentropy is almost always the loss function we should use for such problems. It minimizes the distance between the probability distributions output by the network and the true distribution of the targets.

6. There are two ways to handle labels in multiclass classification:
Encoding the labels via categorical encoding (also known as one-hot encoding) and using categorical_crossentropy as a loss function Encoding the labels as integers and using the sparse_categorical_crossentropy loss function

7. If we need to classify data into a large number of categories, we should avoid creating information bottlenecks in our network due to intermediate layers that are too small.


## Some Neural Regression

We have done a lot of classification, now let's do a bit of regression using the **Boston Housing Dataset**.

Target: Predicting the price from a given dataset of old house prices.

For the particular _Boston Housing_ dataset, the values of different attributes vary wildly in their range and this makes it very hard to generalize as the parameters have to change a lot and even for the smallest of changes the outcome varies absurdly. Hence to curb this situation, the real-valued data is normalized so that a standardize distribution can be attained all over the distribution and thus making it easy to generalize.

Generally, to normalize one can subtract mean of the particular attribute and divide it by the variance of the attribute. This gives the attribute mean of zero and variance of 1.

```python
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
```

### Things to remember

1. Regression is done using different loss functions than what we used for classification. Mean squared error ( `MSE` ) is a loss function commonly used for regression.

2. Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally, the concept of accuracy doesn’t apply for regression. A common regression metric is mean absolute error (`MAE` ).

3. When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step.

4. When there is little data available, using K-fold validation is a great way to reliably evaluate a model.

5. When little training data is available, it’s preferable to use a small network with few hidden layers (typically only one or two), in order to avoid severe overfitting.


# Deep Learning Trivia

## Supervised learning

This is by far the most common case. It consists of learning to map input data to known targets (also called annotations), given a set of examples (often annotated by humans).
All four examples we’ve encountered so far were canonical
examples of supervised learning. Generally, almost all applications of deep learning that are in the spotlight these days belong in this category, such as optical character recognition, speech recognition, image classification, and language translation. Although supervised learning mostly consists of classification and regression, there
are more exotic variants as well, including the following (with examples):

- **Sequence generation**: Given a picture, predict a caption describing it. Sequence generation can sometimes be reformulated as a series of classification problems (such as repeatedly predicting a word or token in a sequence).

- **Syntax tree prediction**: Given a sentence, predict its decomposition into a syntax
tree.

- **Object detection**: Given a picture, draw a bounding box around certain objects inside the picture. This can also be expressed as a classification problem (given many candidate bounding boxes, classify the contents of each one) or as a joint classification and regression problem, where the bounding-box coordinates are
predicted via vector regression.

- **Image segmentation**: Given a picture, draw a pixel-level mask on a specific object.


## Unsupervised Learning

This branch of machine learning consists of finding interesting transformations of the input data without the help of any targets, for the purposes of data visualization, data compression, or data denoising, or to better understand the correlations present in the data at hand. Unsupervised learning is the bread and butter of data analytics, and it’s often a necessary step in better understanding a dataset before attempting to solvea supervised-learning problem. Dimensionality reduction and clustering are well-known categories of unsupervised learning.

## Self- Supervised learning

This is a specific instance of supervised learning, but it’s different enough that it deserves its own category. Self-supervised learning is supervised learning without human-annotated labels, we can think of it as supervised learning without any humans in the loop. There are still labels involved (because the learning has to be supervised by something), but they’re generated from the input data, typically using a heuristic algorithm. For instance, **_Autoencoders_** are a well-known instance of self-supervised learning, where the generated targets are the input, unmodified. In the same way, trying to predict the next frame in a video, given past frames, or the next word in a text, given previous words, are instances of self-supervised learning (temporally supervised learning, in this case: supervision comes from future input data)


## Reinforcement Learning

Long overlooked, this branch of machine learning recently started to get a lot of attention after Google DeepMind successfully applied it to learning to play Atari games (and, later, learning to play Go at the highest level). In reinforcement learning, an agent receives information about its environment and learns to choose actions that will maximize some reward. For instance, a neural network that “looks” at a video-game screen and outputs game actions in order to maximize its score can be trained via reinforcement learning

**Find out more about train-val-test splits [here](https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set)** _make sure we understand it_


## Data preprocessing, feature engineeing and feature learning

### Data Prep(processing)

**Vectorization**
All inputs and targets in a neural network must be tensors of floating-point data (or, in specific cases, tensors of integers). Whatever data we need to process sound, images, text we must first turn into tensors, a step called data vectorization. For instance, in the two previous text-classification examples, we started from text represented as lists of integers (standing for sequences of words), and we used one-hot encoding to turn them into a tensor of float32 data. In the examples of classifying digits and predicting house prices, the data already came in vectorized form, so we were able to skip this step.

Then there are all sort of fancy things:

- Data normalization (Makes data _homo_-genius [homogenous])
- Handling Missing values
- Making sense of problem!

### Feature engineering

Feature engineering is the process of using our own knowledge about the data and about the machine-learning algorithm at hand (in this case, a neural network) to make the algorithm work better by applying
hardcoded (nonlearned) transformations to the data before it goes into the model. In many cases, it isn’t reasonable to expect a machine-
learning model to be able to learn from completely arbitrary data. The data needs to be presented in a way that will make the model's job easier.

_Before deep learning, feature engineering used to be critical, because classical shallow algorithms didn’t have hypothesis spaces rich enough to learn useful features by themselves. The way we presented the data to the algorithm was essential to its success. For instance, before convolutional neural networks became successful on the MNIST digit-classification problem, solutions were typically based on hardcoded features such as the number of loops in a digit image, the height of each digit in an image, a histogram of pixel values, and so on._

**Good features still allow we to solve problems more elegantly while using fewer resources. For instance, it would be a terrible idea if we want to use deep learning to make a picture black and white (smh!)**

### Read about the classic learning problem of overfitting and underfitting [here](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76)


## Um-Hmm, THE UNIVERSAL FLOW OF MACHINE LEARNING

1. First, we must define the problem at hand:

  - What will our input data be? What are we trying to predict? we can only learn to predict something if we have available training data: for example, we can only learn to classify the sentiment of movie reviews if we have both movie reviews and sentiment annotations available. As such, data availability is usually the limiting factor at this stage (unless we have the means to pay people to collect data for we).
  - What type of problem are we facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Something else, like clustering, generation, or reinforcement learning? Identifying the problem type will guide our choice of model architecture, loss function, and so on.

**we can’t move to the next stage until we know what our inputs and outputs are, and what data we’ll use**

_Attention_: One class of unsolvable problems we should be aware of is nonstationary problems. Suppose we’re trying to build a recommendation engine for clothing, we’re training it on one month of data (August), and we want to start generating recommendations in the winter. One big issue is that the kinds of clothes people buy change from season to season: clothes buying is a nonstationary phenomenon over the scale of a few months. What we’re trying to model changes over time. In this case, the right move is to constantly retrain our model on data from the recent past, or gather data at a timescale where the problem is stationary. For a cyclical problem like clothes buying, a few years’ worth of data will suffice to capture seasonal variation—but remember to make the time of the year an input of our model! **FEATURE ENGINEERING**


2. Choosing a measure: LOSS/METRIC FUNCTIONS

To control something, we need to be able to observe it. To achieve success, we must define what we mean by success - `accuracy`? `Precision` and `recall`? Customer-retention rate? our metric for success will guide the choice of a `loss function`: what our model will optimize. It should directly align with our higher-level goals, such as the success of our business. For balanced-classification problems, where every class is equally likely, accuracy and area under the receiver operating characteristic curve ( `ROC AUC` ) are common metrics. For class-imbalanced problems, we can use `precision` and `recall`. For ranking problems or multilabel classification, we can use mean average precision. And it isn’t uncommon to have to define our own custom metric by which to measure success

3. Deciding on an evaluation function

Once we know what we’re aiming for, we must establish how we’ll measure our current progress. Commonly used,

  - Maintaining a hold-out validation set—The way to go when we have  plenty of data
  - Doing K-fold cross-validation—The right choice when we have too few samples for hold-out validation to be reliable
  - Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available

4. Preparing the Dataset

Once we know what we’re training on, what we’re optimizing for, and how to evaluate our approach, we’re almost ready to begin training models. But first, we should format our data in a way that can be fed into a machine-learning model.

5. Developing a model that does better than a baseline

This is the area where our talent must shine, either it be a research project or some hack it should perform at par or better than a baseline.
But, note that it’s not always possible to achieve statistical power. If we can’t beat a random baseline after trying multiple reasonable architectures, it may be that the answer to the question we’re asking isn’t present in the input data. Remember that we make two hypotheses:

  - We hypothesize that our outputs can be predicted given our inputs.
  - We hypothesize that the available data is sufficiently informative to learn the relationship between inputs and outputs

It may well be that these hypotheses are false, in which case we must go back to the drawing board.
Assuming that things go well, we need to make three key choices to build our first working model:

  - Last-layer activation—This establishes useful constraints on the network’s output. For instance, the IMDB classification example used sigmoid in the last layer the regression example didn’t use any last-layer activation; and so on.
  - Loss function—This should match the type of problem we’re trying to solve. For instance, the IMDB example used binary_crossentropy , the regression example used mse , and so on.
  - Optimization configuration—What optimizer will we use? What will its learning rate be? In most cases, it’s safe to go with rmsprop and its default learning rate.

_Regarding the choice of a loss function, note that it isn’t always possible to directly optimize for the metric that measures success on a problem. Sometimes there is no easy way to turn a metric into a loss function; loss functions, after all, need to be computable given only a mini-batch of data (ideally, a loss function should be computable for as little as a single data point) and must be differentiable (otherwise, we can’t use backpropagation to train your network). For instance, the widely used classification metric ROC AUC can’t be directly optimized. Hence, in classification tasks, it’s common to optimize for a proxy metric of ROC AUC , such as crossentropy. In general, we can hope that the lower the crossentropy gets, the higher the ROC AUC will be._

6. Scaling up: Develop a model that Overfits!

What?

7. Regularize your model to curb the overfitting with the classic method

Ohhhh!! Makes sense! To overfit a model is easy as you've to increase the complexity only, (turning up the layers) once we get a model that overfits, we can easily regularize the model to reduce the overfitting.
Some methods are -
  - Add dropout.
  - Try different architectures: add or remove layers.
  - Add L1 and/or L2 regularization.
  - Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.
  - Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.

# The Process in short:

1. Define the problem
2. Loss/metric function
3. Evaluation
4. Preparing the Dataset
5. Defeat baseline
6. Overfit
7. Regularize
