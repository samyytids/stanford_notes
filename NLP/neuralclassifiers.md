word2vec is a bag of words model, meaning that it doesn't care about proximity and order.
Basic gradient descent is all well and good, but it is somewhat outdated. So, we shouldn't use it in practice.
![[Pasted image 20231011185407.png]]
Reason for this is that we are calculating the gradients with respect to 1 variable at a time. So, in the case of NLP problems we may have 9238472938472390847 words in our corpus and thus we would need to perform gradient descent for each and every one of them. This could take a LONG time. 

We instead use stochastic gradient descent. 
This is the same principle as standard gradient descent, but we just use a random sample (this is a static random sample, so the first choice is repeated over and over again) of our corpus to estimate the gradients. This leads to noisy gradients but massively reduces processing time. 
Which I suppose balances itself out. Interestingly apparently stochastic random descent performs better in terms of speed AND accuracy, despite being inherently noisier than standard gradient descent. 

Alternative to the naive softmax we were looking at in [Lecture 1 -wordvectors](wordvector.md) which is computationally expensive because the denominator requires computing a sum for each element in our corpus. We can use skip gram model with negative sampling. 

## Skip gram with negative sampling
Basic idea is you train two binary logistic regressions. 
One that uses a true pair (centre word and a real context word) and another that uses several noise pairs (centre word and a random word from the corpus)

The logic is still minimising the average loss for each given centre word:
![[Pasted image 20231011190413.png]]

![[Pasted image 20231011190459.png]]

But we apply it to this function instead where we seek to maximise the above function
the logsigma() in this case is the logistic function which maps a number to a probability. The left log is the probability transofmation of the dot product between our centre word and a true pair while the right log is the sum of the probability of our centre word and all of the noise pairs. 
The reason why we are using the negative of the dot product on the right is so that we can use the symmetric nature of the logistic distribution and replace small numbers with big numbers. 
![[Pasted image 20231011190823.png]]
Above is a transformation of the objective function of Skip gram that makes it align with our usual assumption of wanting to reduce the objective function rather than maximise it. 
k being our negative sampling where we take K negative samples (using word probabilities). Negative sampling referring to the fact that these are your "negative words" aka your bullshit terms. 
Another trick used here is how they sample words from the corpus, they use a *unigram distribution* which takes the amount of times a word occurs in the corpus and raises it to the $\frac{3}{4}$ power (this dampens the extreme ends of the distribution) we then divide that by the total of number of occurrences  in our corpus. 
![[Pasted image 20231011191211.png]]


Alternatives:

## Co-occurrence matrix
Provide a window we want to check and then note down the amount of times that a given series of words occur together. 
eg with a window of 1, centre +/- 1.
We count each time that a particular pair of words occur and plot them in a matrix.
![[Pasted image 20231011191444.png]]
Can be applied to windows like word2vec, or it can be used on whole documents or arbitrary document segments. 

#### Issues
Usually create huge high dimensional vectors that are very sparse and very noisy. 

As such it's generally better to use a dense low dimensional vector. 

How can we reduce the dimensionality?
Singular value decomposition:
	You take 1 matrix and convert it into 3 matrices. 
	![[Pasted image 20231011192733.png]]
	the shaded sections are sections that are effectively dead. 
	the sigma matrix has values in descending importance, so if you want to further reduce the dimensionality you can remove the lower values from the sigma matrices. (This will "kill" stuff in the V transpose matrix and our U matrix.)
	![[Pasted image 20231011192938.png]]
	SVD doesn't work very well in this scenario however, do to SVD using the good ol' normally distributed errors (which is very unlikely to be the case in a sparse matrices like the co-occurrence matrix).
	We also have the issue of some uninteresting words getting amplified because they are common words (pronouns).
	We can somewhat adjust for this by using some sort of scaling on the counts. 
	log the counts
	set a maximum count
	ignore function words (I assume this is like cutting out stop words).
	Looking back at the analogy stuff in [Lecture 1 -wordvectors](wordvector.md) we can also use co-occurrence matrices for this task, but you will want to use the ratio of probabilities rather than the straight probabilities due to quirks with semantics. 
	![[Pasted image 20231011193606.png]]
	

###  Evaluating word vectors
2 general ways of evaluation within NLP
#### Intrinsic
evaluation on a specific/intermediate subtask
fast to compute
helps to understand that system 
not necessarily going to carry over to the real task unless both tasks are highly correlated
Think of this as looking at a specific section of a potential real task

#### Extrinsic
evaluation on a real task 
longer to compute accurately
unclear if the subsystem is the problem or its interaction or other subsystems
if replacing exactly 1 subsystem with another improves accuracy then we happy
think of this as looking at throwing stuff at a real task and trying to be as granular as possible with what you change. 


It appears that in general a vector size of 300 seems to be the sweet spot for diminishing returns. 

## How do we handle multi-meaning words
Since we have one vector for each word how do we handle words that have numerous different vectors? 
We can use word sense vectors (where you have a vector for each separate version of the word).
Which does work. But I imagine this can be sensitive to the amount that a specific meaning appears or if none of them appear often enough to meaningfully discern them?
Due to the short comings of this approach we typically forgo multiple vectors and instead use 1 vector, which becomes the "super position" of that word, basically meaning a weighted sum of the various definitions of the word. As such we end up with the most comprehensive meaning is the most potent one. By using context clues we end up with very effective predictive power despite using the super position. 
Despite the ambiguity of the different meanings and which one is driven by what you can use concepts of sparse coding to extract the specific parts that link to the specific meanings. 