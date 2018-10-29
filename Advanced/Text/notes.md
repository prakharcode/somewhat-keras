# Somewhat Keras
## Text and Sequences

This part of the blog will explore the text and sequencial data processed through some of the most powerful deep learning models.

The two fundamental deep-learning algorithms for sequence processing
are recurrent neural networks and 1D convnets.

Applications of these algorithms include the following:

- Document classification and timeseries classification, such as identifying the topic of an article or the author of a book

- Timeseries comparisons, such as estimating how closely related two documents or two stock tickers are

- Sequence-to-sequence learning, such as decoding an English sentence into French

- Sentiment analysis, such as classifying the sentiment of tweets or movie reviews as positive or negative

- Timeseries forecasting, such as predicting the future weather at a certain location, given recent weather data

## Working with text data

Text is one of the most abundant sequencial data. It can be understood as either a sequence of characters or a sequence of words, but it's most common to work at the level of words. The deep learning sequence processing models that we'll use can use text to produce a basic form of understanding, sufficient for application including document classification, sentiment analysis, author identification and even question-answering(QA).

Deep learning for natural-language processing is pattern recognition applied to words, sentences, and paragraphs, in much the same way that computer vision is pattern recognition applied to pixels.

Like all other neural networks, deep-learning models don’t take as input raw text: they only work with numeric tensors. Vectorizing text is the process of transforming text into numeric tensors. This can be done in multiple ways:
- Segment text into words, and transform each word into a vector.
- Segment text into characters, and transform each character into a vector.
- Extract n-grams of words or characters, and transform each n-gram into a vector. N -grams are overlapping groups of multiple consecutive words or characters.

Collectively, the different units into which you can break down text (words, characters, or n-grams) are called tokens, and breaking text into such tokens is called tokenization. All text-vectorization processes consist of applying some tokenization scheme and then associating numeric vectors with the generated tokens.

These vectors, packed into _sequence tensors_, are fed into deep neural networks. There are multiple ways to associate a __vector__ with a token. In this section, I’ll present two major ones: __one-hot encoding__ of tokens, and __token embedding__ (typically used exclusively for words, and called __word embedding__).

Articles on n-grams and bag of words. [Here]() and [here]().


## One-hot encoding of words

One-hot encoding is the most common, most basic way to turn a token into a vector.

It consists of associating a unique integer index with every word and then turning this integer index i into a binary vector of size N (the size of the vocabulary); the vector is all zeros except for the i th entry, which is 1.
_Of course, one-hot encoding can be done at the character level, as well_

## Using word embeddings

Another popular and powerful way to associate a vector with a word is the use of dense word vectors, also called word embeddings. Whereas the vectors obtained through one-hot encoding are binary, sparse (mostly made of zeros), and very high-dimensional (same dimensionality as the number of words in the vocabulary), word embeddings are low dimensional floating-point vectors.

Unlike the word vectors obtained via one-hot encoding, word
embeddings are learned from data. It’s common to see word embeddings that are 256-dimensional, 512-dimensional, or 1,024-dimensional when dealing with very large vocabularies. On the other hand, one-hot encoding words generally leads to vectors that are 20,000-dimensional or greater (capturing a vocabulary of 20,000 tokens, in this case). So, word embeddings pack more information into far fewer dimensions. Less dimensions means lower impact of the curse of dimensionality.

There are two ways to obtain word embeddings:
- Learn word embeddings jointly with the main task you care about (such as document classification or sentiment prediction). In this setup, you start with random word vectors and then learn word vectors in the same way you learn the weights of a neural network.
- Load into your model word embeddings that were precomputed using a different machine learning task than the one you’re trying to solve. These are called pretrained word embeddings. ( This is now referred as transfer learning by some researchers)

### Learning word embeddings with the embedding layer [code]()

The simplest way to associate a dense vector with a word is to choose the vector at random. The problem with this approach is that the resulting embedding space has no structure: for instance, the words accurate and exact may end up with completely different embeddings, even though they’re interchangeable in most sentences. It’s difficult for a deep neural network to make sense of such a noisy, unstructured embedding space.

To get a bit more abstract, the geometric relationships between word vectors should reflect the semantic relationships between these words. Word embeddings are meant to map human language into a geometric space. For instance, in a reasonable embedding space, you would expect synonyms to be embedded into similar word vectors; and in general, you would expect the geometric distance (such as L2 distance) between any two word vectors to relate to the semantic distance between the associated words (words meaning different things are embedded at points far away from each other, whereas related words are closer). In addition to distance, you may want specific directions in the embedding space to be meaningful.

Is there some ideal word-embedding space that would perfectly map human language and could be used for any natural-language-processing task? Possibly, but we have yet to compute anything of the sort. Also, there is no such a thing as human language there are many different languages, and they aren’t isomorphic, because a language is the reflection of a specific culture and a specific context.

But more pragmatically, what makes a good word-embedding space depends heavily on your task:
The perfect word-embedding space for an English-language movie-review sentiment analysis model may look different from the perfect embedding space for an English language legal-document-classification model, because the importance of certain semantic relationships varies from task to task.

It’s thus reasonable to learn a new embedding space with every new task. Fortunately, backpropagation makes this easy, and Keras makes it even easier. It’s about learning the weights of a layer: the Embedding layer.

The Embedding layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, it looks up these integers in an internal dictionary, and it returns the associated vectors. It’s effectively a dictionary lookup

Word index -------> Embedding layer -----> Corresponding word vector

The Embedding layer takes as input a 2D tensor of integers, of shape (samples, sequence_length) , where each entry is a sequence of integers. It can embed sequences of variable lengths: for instance, you could feed into the Embedding layer in the previous example batches with shapes (32, 10) (batch of 32 sequences of length 10) or (64, 15) (batch of 64 sequences of length 15). All sequences in a batch must have the same length, though (because you need to pack them into a single tensor), so sequences that are shorter than others should be padded with zeros, and sequences that are longer should be truncated.

This layer returns a 3D floating-point tensor of shape (samples, sequence_length, embedding_dimensionality) . Such a 3D tensor can then be processed by an RNN layer or a 1D convolution layer (both will be introduced in the following sections).

When you instantiate an Embedding layer, its weights (its internal dictionary of token vectors) are initially random, just as with any other layer. During training, these word vectors are gradually adjusted via backpropagation, structuring the space into something the downstream model can exploit. Once fully trained, the embedding space will show a lot of structure—a kind of structure specialized for the specific problem for which you’re training your model.

Let’s apply this idea to the IMDB movie-review sentiment-prediction task that we’re already familiar with. First, we’ll quickly prepare the data. We’ll restrict the movie reviews to the top 10,000 most common words and cut off the reviews after only 20 words. The network will learn 8-dimensional embeddings for each of the 10,000 words, turn the input integer. sequences (2D integer tensor) into embedded sequences ( 3D float tensor), flatten the tensor to 2D, and train a single Dense layer on top for classification. [code]().


## Using Pre trained word embedding

Sometimes, you have so little training data available that you can’t use your data alone to learn an appropriate task-specific embedding of your vocabulary. What do you do then?
Instead of learning word embeddings jointly with the problem you want to solve, you can load embedding vectors from a precomputed embedding space that you know is highly structured and exhibits useful properties—that captures generic aspects of language structure. The rationale behind using pretrained word embeddings in natural-language processing is much the same as for using pretrained convnets in image classification: you don’t have enough data available to learn truly powerful features on your own, but you expect the features that you need to be fairly
generic—that is, common visual features or semantic features. In this case, it makes sense to reuse features learned on a different problem.

The idea of a dense, low-dimensional embedding space for words, computed in an unsupervised way, was initially explored by Bengio et al. in the early 2000s, but it only started to take off in research and industry applications after the release of one of the most famous and successful word-embedding schemes: the Word2vec algorithm (https://code.google.com/archive/p/word2vec), developed by Tomas Mikolov at Google in 2013. Word2vec dimensions capture specific semantic properties, such as gender.

There are various precomputed databases of word embeddings that you can download and use in a Keras Embedding layer. Word2vec is one of them. Another popular one is called Global Vectors for Word Representation (GloVe, https://nlp.stanford.edu/projects/glove), which was developed by Stanford researchers in 2014. This embedding technique is based on factorizing a matrix of word co-occurrence statistics. Its developers have made available precomputed embeddings for millions of English tokens, obtained from Wikipedia data and Common Crawl data.

### Doing it from raw text.
[code]()

- Download the data
- Tokenize the data
- Download Glove embeddings
- Preprocess the embeddings
- Define a model
- Load the Glove embeddings
- Training and evaluating

The model quickly starts overfitting, which is unsurprising given the small number of training samples. Validation accuracy has high variance for the same reason, but it seems to reach the high 50s.

Note that our mileage may vary: because you have so few training samples, performance is heavily dependent on exactly which 200 samples we choose—and you’re choosing them at random. If this works poorly for we, try choosing a different random set of 200 samples, for the sake of the exercise (in real life, we don’t get to choose your training data).

Training the model without pretrained word embeddings and using the random embedding layer. [code]()

## Understandng Recurrent Neural Networks

A major characteristic of all neural networks you’ve seen so far, such as densely connected networks and convnets, is that they have no memory. Each input shown to them is processed independently, with no state kept in between inputs. With such networks, in order to process a sequence or a temporal series of data points, you have to show the entire sequence to the network at once: turn it into a single data point. For instance, this is what you did in the IMDB example: an entire movie review was transformed into a single large vector and processed in one go. Such networks are called feedforward networks.

In contrast, as you’re reading the present sentence, you’re processing it word by word—or rather, eye saccade by eye saccade—while keeping memories of what came before; this gives you a fluid representation of the meaning conveyed by this sentence. Biological intelligence processes information incrementally while maintaining an internal model of what it’s processing, built from past information and constantly updated as new information comes in.

A recurrent neural network ( RNN ) adopts the same principle, albeit in an extremely simplified version: it processes sequences by iterating through the sequence elements and maintaining a state containing information relative to what it has seen so far. In effect, an RNN is a type of neural network that has an internal loop (see figure 6.9).
The state of the RNN is reset between processing two different, independent sequences (such as two different connection IMDB reviews), so you still consider one sequence a single data point: a single input to the network. What changes is that this data point is no longer processed in a  single step; rather, the network internally loops over sequence elements

To make these notions of loop and state clear, let’s implement the forward pass of a toy RNN in Numpy. This RNN takes as input a sequence of vectors, which you’ll encode as a 2D tensor of size (timesteps, input_features) . It loops over timesteps, and at each timestep, it considers its current state at t and the input at t (of shape (input_ features,) , and combines them to obtain the output at t . You’ll then set the state for the next step to be this previous output. For the first timestep, the previous output isn’t defined; hence, there is no current state. So, you’ll initialize the state as an all zero vector called the initial state of the network. [code]()

Easy enough: in summary, an RNN is a for loop that reuses quantities computed during the previous iteration of the loop, nothing more. Of course, there are many different RNN s fitting this definition that you could build—this example is one of the simplest RNN formulations

```python
output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
```

Simple implementation of RNN using Keras [code]().

## Understandng the LSTM and GRU layers

SimpleRNN isn’t the only recurrent layer available in Keras. There are two others: LSTM and GRU . In practice, you’ll always use one of these, because SimpleRNN is generally too simplistic to be of real use. SimpleRNN has a major issue: although it should theoretically be able to retain at time `t` information about inputs seen many timesteps before, in practice, such long-term dependencies are impossible to learn. This is due to the vanishing gradient problem, an effect that is similar to what is observed with non-recurrent networks (feedforward networks) that are many layers deep: as you keep adding layers to a network, the network eventually becomes untrainable. The theoretical reasons for this effect were studied by Hochreiter, Schmidhuber, and Bengio in the early 1990s. The LSTM and GRU layers are designed to solve this problem.

**How is LSTM/GRU better than Simple RNN? [read here.]()**

## LSTM in Keras

We’ll set up a model using an LSTM layer and train it on the IMDB data. The network is similar to the one with SimpleRNN that was just presented. You only specify the output dimensionality of the LSTM layer; leave every other argument (there are many) at the Keras defaults. Keras has good defaults, and things will almost always “just work” without you having to spend time tuning parameters by hand

[Here is the code.]()

## Advanced use of recurrent neural network

In this section, we’ll review three advanced techniques for improving the performance and generalization power of recurrent neural networks. By the end of the section, you’ll know most of what there is to know about using recurrent networks with Keras. We’ll demonstrate all three concepts on a temperature-forecasting problem, where we have access to a timeseries of data points coming from sensors installed on the roof of a building, such as temperature, air pressure, and humidity, which you use to predict what the temperature will be 24 hours after the last data point. This is a fairly challenging problem that exemplifies many common difficulties encountered when working with timeseries.

We'll learn about:

- Recrrent Dropout
- Stacking recurrent layers
-  Bidirectional recurrent layers

### A temprature forcasting problem

Until now, the only sequence data we’ve covered has been text data, such as the IMDB dataset and the Reuters dataset. But sequence data is found in many more problems than just language processing. In all the examples in this section, you’ll play with a weather timeseries dataset recorded at the Weather Station at the Max Planck Institute for Biogeochemistry in Jena, Germany. 4

In this dataset, 14 different quantities (such air temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. The original data goes back to 2003, but this example is limited to data from 2009–2016. This dataset is perfect for learning to work with numerical timeseries. We’ll use it to build a model that takes as input some data from the recent past (a few days’ worth of data points) and predicts the air temperature 24 hours in the future.

Download and uncompress the data as follows:

```bash
cd ~/Downloads
mkdir jena_climate
cd jena_climate
wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
unzip jena_climate_2009_2016.csv.zip
```

### Preparing the data

The exact formulation of the problem will be as follows: given data going as far back as lookback timesteps (a timestep is 10 minutes) and sampled every steps timesteps, can you predict the temperature in delay timesteps? We’ll use the following parameter values:
(These numbers are the number of rows in the timeseries table)

- lookback = 720 —Observations will go back 5 days. (5 days * 24 Hours * 60 mins / 10 mins = 720 steps)

- steps = 6 —Observations will be sampled at one data point per hour. (10 mins * 6 steps = 60 mins = 1 hour)

- delay = 144 —Targets will be 24 hours in the future. (24 hours * 60 mins / 10 mins = 144 steps)

To get started, you need to do two things:

- Preprocess the data to a format a neural network can ingest. This is easy: the data is already numerical, so you don’t need to do any vectorization. But each timeseries in the data is on a different scale (for example, temperature is typically between -20 and +30, but atmospheric pressure, measured in mbar, is around 1,000). We’ll normalize each timeseries independently so that they all
take small values on a similar scale.

- Write a Python generator that takes the current array of float data and yields batches of data from the recent past, along with a target temperature in the future. Because the samples in the dataset are highly redundant (sample N and sample N + 1 will have most of their timesteps in common), it would be wasteful to explicitly allocate every sample. Instead, we’ll generate the samples on the fly using the original data.

We’ll preprocess the data by subtracting the mean of each timeseries and dividing by the standard deviation. We’re going to use the first 200,000 timesteps as training data, so compute the mean and standard deviation only on this fraction of the data.


It yields a tuple (samples, targets), where samples is one batch of input data and targets is the corresponding array of target temperatures. It takes the following arguments:

- data —The original array of floating-point data, which you normalized in listing 6.32.
- lookback —How many timesteps back the input data should go.
- delay —How many timesteps in the future the target should be.
- min_index and max_index —Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
- shuffle —Whether to shuffle the samples or draw them in chronological order.
- batch_size —The number of samples per batch.
- step —The period, in timesteps, at which you sample data. You’ll set it to 6 in order to draw one data point every hour.

### A common-sense, non machine learning baseline

### A basic machine-learning approach

### A first recurrent baseline

### Using recurrent dropout to fight overfitting

### Stacking Recurrent Layers

### Using Bidirectional RNNs

## Sequence Processing with 1D Convnet

The convolution layers introduced previously were 2D convolutions, extracting 2D patches from image tensors and applying an identical transformation to every patch.
In the same way, you can use 1D convolutions, extracting local 1D patches (subsequences) from sequences.
Such 1D convolution layers can recognize local patterns in a sequence. Because the same input transformation is performed on every patch, a pattern learned at a certain position in a sentence can later be recognized at a different position, making 1D conv- nets translation invariant (for temporal translations). For instance, a 1D convnet processing sequences of characters using convolution windows of size 5 should be able to learn words or word fragments of length 5 or less, and it should be able to recognize these words in any context in an input sequence. A character-level 1D convnet is thus able to learn about word morphology.

### Implementation of conv 1d for sequence data

1D convnets are structured in the same way as their 2D counterparts: they consist of a stack of Conv1D and MaxPooling1D layers, ending in either a global pooling layer or a Flatten layer, that turn the 3D outputs into 2D outputs, allowing you to add one or more Dense layers to the model for classification or regression.

One difference, though, is the fact that you can afford to use larger convolution windows with 1D convnets. With a 2D convolution layer, a 3 × 3 convolution window contains 3 × 3 = 9 feature vectors; but with a 1D convolution layer, a convolution window of size 3 contains only 3 feature vectors. You can thus easily afford 1D convolution windows of size 7 or 9.

[Check out the code]().

## Combining Cnn and Rnn for long sequences

Because 1D convnets process input patches independently, they aren’t sensitive to the order of the timesteps (beyond a local scale, the size of the convolution windows), unlike RNN s. Of course, to recognize longer-term patterns, you can stack many convolution layers and pooling layers, resulting in upper layers that will see long chunks of the original inputs - but that’s still a fairly weak way to induce order sensitivity. One way to evidence this weakness is to try 1D convnets on the temperature-forecasting problem, where order-sensitivity is key to producing good predictions. The following example reuses the following variables defined previously: `float_data` , `train_gen`, `val_gen` , and `val_steps`.

[code]().
