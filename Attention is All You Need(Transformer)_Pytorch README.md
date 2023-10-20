# Attention is All You Need

Authors : Ashish Vaswani et al.

Date of Publish : 12 Jun 2017

## About Attention is All You Need

- introduces a new simple network architecture, the ‘Transformer’ based solely on attention mechanisms.
- the Transformer allowed significantly more parallelization and reached a new state of the art in translation quality.

## Model Architecture

<img width="637" alt="Untitled 0" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/d1509f78-c1e1-4c16-a631-71bbfe63a96e">

- the model is composed of Encoder and Decoder.
- the encoder maps an input sequence of symbol representations to a sequence of continuous representations.
- the decoder generates an output sequence of symbols when given the output of the encoder, one at a time.

### Scaled Dot-Product Attention

 

<img width="389" alt="Untitled 1" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/83bd68cc-415f-4995-94a5-546f277ba858">

- dot products of the query with all keys is computed, then each divided by square root of dimension of key to scale.
- after, softmax function is applied to obtain the weights on the values, then matrix multiplication with the values is followed.

<img width="939" alt="Untitled 2" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/125dfd42-b1db-49be-a202-b905a1c8b165">

- **Why scale?** : According to the paper, for large values of dimension of keys, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

### Multi-Head Attention

<img width="474" alt="Untitled 3" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/dcf0e938-51fb-4760-8f65-99e3f9dfcb4d">

- linearly projecting the queries, keys, and values h times with different, learned linear projections to query dimension, key dimension, and value dimensions.
- the output values of h scaled dot-product attentions are concatenated and then projected.

 

<img width="990" alt="Untitled 4" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/3332b634-ec99-4abd-b44c-c1a240ef7748">

- **why multi-head attention?** : allows the model to jointly attend to information from different representation subspaces at different positions.

### Applications of Attention

**1) self-attention layer(in encoder) :** all of the keys, values, and queries comes from the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. 

**2) self-attention layer(in decoder, masked) :** all of the keys, values, and queries comes from the output of the previous layer in the decoder. Each position in the decoder can attend to all positions in the decoder up to and including that position. But, the future tokens are masked out while training to prevent leftward information flow. 

- **why self-attention? :**
    - 1) total computational complexity per layer (smaller than other networks)
    - 2) amount of computation that can be parallelized (higher than other networks)
    - 3) path length between long-range dependencies in the network is short, which makes it easier for the network to learn long-range dependencies.
        
        <img width="946" alt="Untitled 5" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/8f0d42d6-aa1d-4709-92bc-0c9f9771dc92">
        

**3) encoder-decoder attention layer :** the queries come from the previous decoder layer, and the keys and values come from the output of the encoder. Every position in the decoder can attend over all positions in the input sequence. 

### Pointwise Feed Forward Network

- consists of two linear transformations with a ReLU activation in between.
- In the paper, the inner-layer has dimensionality of 2048, and the input/output layer has dimensionality of 512.

<img width="983" alt="Untitled 6" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/e904e7fd-cd13-44da-86b5-920ed829e9d4">

### Encoder

- Encoder is composed of a stack of N identical layers.
- Each layer has two sub-layers: first is multi-head self-attention mechanism, and second is positionwise fully connected feed-forward network.
- residual connection around each of the two sub-layers are employed, and then followed by layer normalization.
- In the paper, to facilitate these residual connections, all sub-layers in the model are 512 dimensions.

### Decoder

- Decoder is composed of a stack of N identical layers.
- Each layer has three sub-layers: first is masked multi-head self-attention, second is encoder-decoder attention layer, and the third is positionwise fully connected feed-forward network.
- residual connection around each of the three sub-layers are employed, and then followed by layer normalization.

### Embedding and Positional Encoding

- In the embedding layers, the weights are multiplied by square root of the dimension of the model.
- positional encoding is added to inject information about the relative or absolute position of the tokens in the sequence(the model does not contain recurrence or convolution).
- positional encoding has the same dimension as the embeddings(dimension of the model), so that the positional encoding can be summed to the token embeddings.
- In the paper, sine and cosine functions are used for the positional encoding, where pos is the position and i is the dimension.

<img width="1059" alt="Untitled 7" src="https://github.com/aerojohn1223/NLPModels-byPytorch/assets/82106824/dd870c7e-b44a-4c21-bbcd-c9c16b33ca79">

- for my implementation(pytorch), I did not use these functions, and used embeddings that can be learned instead, which does not make any problem according to the paper.

## Implementation

- I’ve tried to implement Transformer network with help
- The reason why I implemented Transformer is because lots of computer vision networks are using vision transformers in their networks.
