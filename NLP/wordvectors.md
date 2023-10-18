
## Introduction to word vectors


Word vectors (embeddings): These are dense vectors that are used to describe the meaning of a word using the concept of distributional semantics. This being that the meaning of a word is created in part by the words that surround it. These vectors are dense vectors of real numbers which detail the relationship between that word and those that are typically found near it. 

Embeddings: We refer to things in feature engineering as embeddings because we are typically creating vectors as ways to represent our data and these vectors embed our data point into a (generally) high dimensional vector space. 

Word2Vec algorithm: 
	Every word is represented by a vector. 
	We go through each word in the text, the centre word C and outside word O are used to create the vector. 
	We use the similarity of the word vectors for C and O to calculate the probability of O given C. 
	We adjust the vectors to maximise this probability. 

![[Pasted image 20231009201600.png]]
L(theta): Our likelihood in terms of our word vector
Product 1: Product of each of our words as the centre word
Product 2: Product of each word in the window around that centre word 
Probability: multiplied by the probability of a given outside word occurring given our centre word. 

In order to learn this probability we need to optimise our loss function. 
The reason why we don't directly maximise our likelihood as our loss function is because it is easier to work with sums than it is to work with products. 
By using the log likelihood we turn all of our products into sums. 

The minus is just so that we are minimising our objective function instead of maximising it (seemingly purely because it stops us from having to remember the direction we are looking for for each possibly loss function)
1/T gives us the average (since we are dividing the likelihood by our corpus size)

How do we calculate $P(W_{t+j}|W_{t};\theta)$?
We use two vectors per word w. 
$v_w$ when W is the centre word .
$u_w$ when W is the context (outside) word. 
Then for a centre word C and a context word O:
$P(o|c) = \frac{exp(u^T_ov_c)}{\sum_{w\in v}exp(u^T_wv_c)}$ 
$exp(u^T_ov_c)$ 
This is the exponent of the dot product of our two vectors for our outside and centre words.
We exponentiate the dot product because exponents cannot be negative and because we are looking for probabilities which are by definition bound by 0 we cannot have a negative result. 
This is used to calculate the similarity of the two words
The denominator is simply used to normalise over the entire vocabulary to give us a probability distribution. 
![[Pasted image 20231009204155.png]]
This translates to a version of a softmax function. 
Softmax gets its name because it "maximises" the probability of the largest things (in this case the largest thing being the most similar things). Soft because it still gives some probability to smaller things. <- the soft part is I assume what makes it return a distribution of items rather than just the max value. 

I am going to skip notes on the proofs since it involves multivariable derivatives but here's the end.
![[Pasted image 20231009205929.png]]
What this basically reduces down to is that the process of gradient descent with a softmax function boils down to being a measure of whether our expectation and our observed reality match up. 
As such we adjust our params as much as possible so that our gradient = 0. 
In this case our params just being our word vectors being messed around with until we get something that accurately predicts our centre word given our outside words and vice versa. 
## Important terms
#Corpus Body of text.


## Links
- [Back to course overview](CS224N.md) 