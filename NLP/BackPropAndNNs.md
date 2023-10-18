### Neural networks by hand

#### Name Entity Recognition
This is a process of classification whereby you classify words in a sentence into groups (these being naming words). 

##### Simple NER
Take a word2vec model, use it in an NN and pass it through a logisitic classifier to give the probability of being in a particular class. 
This is a supervised training task. 

![[Pasted image 20231012214958.png]]

Take our input vector (standard window and centre word approach)

- Apply a vector of weights and a bias to it.
- Then apply the result of that transformation to an activation function.
- This hidden layer will return some hidden vector h. 
- We then get the dot product of h and u (not sure where u comes from yet)
- We finally pass that dot product through our logistic function to get a probability of being some classification. 

For our hand worked example we are going to use a stochastic gradient descent equation for finding our loss. 

![[Pasted image 20231012220617.png]]
![[Pasted image 20231012220640.png]]

For each parameter we are using the partial derivative with respect to that parameter, adjusting that result with our learning rate to find our new value for parameter $\theta_j$. 
(We also apply similar logic to updating our word vectors as well)

##### $\Delta_\theta J(\theta)$ by hand
[If you get confused](book.pdf)
We are going to be doing multi-variable calculus which is just calculus for matrices and vectors

$f(x) = x^3$
$\frac{df}{dx} = 3x^2$

Intuition of a derivative if we change our input a little bit how much will our output change?

If we have multiple inputs then our gradient is a vector of partial derivatives with respect to each input:
$f(x) = f(x_1, x_2,x_3, ..., x_n)$ 
$\frac{\delta f}{\delta x} = \begin{bmatrix}\frac{\delta f}{\delta x_1}, \frac{\delta f}{\delta x_2}, ...., \frac{\delta f}{\delta x_n} \end{bmatrix}$ 

In a NN however we have a function that takes n inputs and provides m outputs
$f(x) = f_1(x_1, x_2, ..., x_n), ..., f_m(x_1, x_2, ..., x_n)$

it's Jacobian is an **m x n** matrix of partial derivatives

$$\frac{\delta f}{\delta x} = \begin{bmatrix}\frac{\delta f_1}{\delta x_1} & \cdots & \frac{\delta f_1}{\delta x_n} \\ \vdots & \ddots & \vdots \\ \frac{\delta f_m}{\delta x_1} & \cdots & \frac{\delta f_m}{\delta x_n}\end{bmatrix}$$
#Jacobian this is just an **m x n** matrix of all the ins and outs of partial derivatives.
$$
\begin{aligned}
z = 3y \\
y = x^2 \\
\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx} = (3)(2x) = 6x
\end{aligned}
$$
Above is a demonstration of the #ChainRule what this basically means is that a function within a function you can find the derivative of one output with the input of any of its sub functions. You just have to do the above. 
You multiply the derivative with respect to the sub function and multiply that by the derivative of the sub function on the input in question.

Using Jacobians:

###### Element-wise activation function

$$
\begin{aligned}
\boldsymbol{h} = f(\boldsymbol{z}) \\
\boldsymbol{z} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b} \\
\frac{\delta \boldsymbol{h}}{\delta \boldsymbol{x}} = \frac{\delta \boldsymbol{h}}{\delta \boldsymbol{z}}\frac{\delta \boldsymbol{z}}{\delta \boldsymbol{x}} = ... \\
\boldsymbol{h} = \text{output of hidden layer} \\
\boldsymbol{z} = \text{vector return from applying weights and biases} \\
\boldsymbol{f()} = \text{our activation function}
\end{aligned}
$$
What the above is basically saying is:
to take our the derivative of our hidden output vector **h** with respect to our input vector **x** we use the same logic as above.
In this case **z** is the vector we get from placing our input vector **x** through a very simple NN layer where we transform it using a vector of weights (**W**) and apply a bias vector **b**.
(remember the $f(z)$ is just our activation function for the hidden layer)

$$
\begin{aligned}
\boldsymbol{h} = f(\boldsymbol{z})\text{, what is }\frac{\delta \boldsymbol{h}}{\delta \boldsymbol{z}}\text{?} & & & & &\boldsymbol{h},\boldsymbol{z} \in \mathbb{R}^n \\
h_i = f(z_i)
\end{aligned} 
$$
What this basically means is how much does a small change in our **z** vector impact our output **h** vector?
In this scenario we have an **n x n** Jacobian. As such our derivative will look like this:
$$
\begin{aligned}
\text{The definition of a Jacobian}\\
(\frac{\delta \boldsymbol{h}}{\delta \boldsymbol{z}})_{ij} = \frac{\delta h_i}{\delta z_j} = \frac{\delta}{\delta z_j}f(z_i)
\end{aligned}
$$
What this basically means is that the elements of the Jacobian are going to take the form of the derivative of each output with respect to each input.
In this instance because this transformation is an element-wise application of an activation function. 

$$
f(z_i) =
\begin{aligned}
\begin{cases}
  f'(z_i)  \text{if } i = j \\
  0 & \text{otherwise} \\
\end{cases} \\
\text{ regular 1-variable derivative}
\end{aligned}
$$
What this basically means is that since we are doing an element-wise thing we only have values when our i = our j. This means that we have a diagonal matrix. ^5873ff

Meaning we end up with the following matrix:

![[Pasted image 20231012230425.png]]
where $f'(z_n)$ is the deriviative of the activation function with input vector $z_n$.  ^c8cf25

###### Basic NN layer
With respect to x.
$$
\frac{\delta}{\delta\boldsymbol{x}}(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}) = \boldsymbol{W}
$$
[explanation](https://www.youtube.com/watch?v=e73033jZTCI) 
Or tldr; it appears that getting the derivative of a matrix functions very similarly to standard derivatives where the following is true 
$$
\begin{aligned}
\frac{\delta}{\delta x} \text{ of } x^TAx = 2Ax \\
\text{Where }x^TAx \text{ is basically the vector equivalent of }Ax^2
\end{aligned}
$$


With respect to b. ^371f8c
$$
\begin{aligned}
\frac{\delta}{\delta\boldsymbol{b}}(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}) = \boldsymbol{I} \text{ the identity matrix} \\
\text{this is basically the same as when get the derivative of a }\\
\text{constant}
\end{aligned}
$$


###### Dot product

$$
\begin{aligned}
\frac{\delta}{\delta\boldsymbol{u}}(\boldsymbol{u}^T\boldsymbol{h}) = \boldsymbol{h}^T \\
\text{Equivalent of deriving x in normal algebra}
\end{aligned}
$$


The transpose is correct, but by the shape convention we instead use **h**. ([[BackPropAndNNs#^1750aa]]) 

# Actually doing the maths!

![[Pasted image 20231012232124.png]]

What we are achieving by finding this derivative is updating our input **b**.
We want to first break down each step as far as possible. 
Here we have a (what I am going to call) a nested layer.
$\boldsymbol{h} = f(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b})$ 
can be decomposed into 
$$
\begin{aligned}
\boldsymbol{h} = f(\boldsymbol{z}) \\
\boldsymbol{z} = \boldsymbol{W}\boldsymbol{x}+\boldsymbol{b}
\end{aligned}
$$
As we saw earlier: [[BackPropAndNNs#^a4fb84]] 

Keep track of dimensionality!

So now that we have the following
$$
\begin{aligned}
s = \boldsymbol{u}^T\boldsymbol{h} &&&&&&&&&&&\frac{\delta s}{\delta \boldsymbol{b}}=\frac{\delta s}{\delta \boldsymbol{h}}\frac{\delta \boldsymbol{h}}{\delta \boldsymbol{z}}\frac{\delta \boldsymbol{z}}{\delta \boldsymbol{b}}\\
\boldsymbol{h} = f(\boldsymbol{z}) \\
\boldsymbol{z} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b} \\
\boldsymbol{x} \text{(input)}
\end{aligned}
$$
Step 1:
	[[BackPropAndNNs#^26e034]]
Step 2:
	[[BackPropAndNNs#^c8cf25]]
Step 3: 
	[[BackPropAndNNs#^371f8c]]
Step 4:
	Multiply them together.
	
	$$
	\begin{aligned}
	\boldsymbol{u}^T\text{diag}(f'(\boldsymbol{z}))\boldsymbol{I} \\
	\boldsymbol{u}^T\circ f'(\boldsymbol{x}) \\
	\end{aligned}
	$$
The $\circ$ is a hadamard product AKA element-wise product.
This can be thought of as naive matrix multiplication basically, take each element of each matrix and multiply them together like you'd expect it to work if you didn't know linear algebra. (comes up a lot in NN)
![[Pasted image 20231012234247.png]]

#### What about with respect to **W**?
Once again we can use the chain rule. 
$\frac{\delta s}{\delta W} = \frac{\delta s}{\delta h}\frac{\delta h}{\delta z}\frac{\delta z}{\delta W}$

May notice something here, this is part of how the backprop algorithm works
Note the first 2 terms in the product with WRT **b** and WRT **W** are identical. As such we can avoid calculating these more than once. 

This is a bite more difficult, we aren't maintaining dimensionality here. We are going from nm inputs to 1 output. 
this causes an issue with out loss function: 
![[Pasted image 20231012235955.png]]
You can't subtract a constant from a matrix. 

So, to avoid this. We move away from pure maths and use the shape convention:  
We basically punch our gradient into the shape of the parameters
![[Pasted image 20231013000136.png]]
$$
\begin{aligned}
\frac{\delta s}{\delta h}\frac{\delta h}{\delta z} = \delta \\
\text{we do this because of the repeating mentioned ealier} \\
\frac{\delta s}{\delta \boldsymbol{W}} = \delta^T\boldsymbol{x}^T
\end{aligned}
$$
$\delta$ is the local error signal at z, x is the local input signal:
but what does that mean?
Local error signal:
	At a particular neuron this represent the error in predicting our output at this step. This is used to adjust parameters in order to reduce the gradient and as such improve our model.

Local input signal:
	This is(These are) the value(s) supplied to our activation function

The reason why both are transpose is the shape convention. By having them both transpose the below happens:
![[Pasted image 20231013000800.png]]


## Back propagation algorithm

This is basically just doing the above, but instead of doing each step over and over we instead find bits that are going to be done repeatedly and effectively cache them. 

#BackwardsPropagation This is just the name for taking values based on our result and updating values in our model. Think of the whole gradient descent process. 
#ForwardsPropagation This is just the process of calculating our end value. 

![[Pasted image 20231013002040.png]]
![[Pasted image 20231013002139.png]]

We have our upstream gradient and we want to get our downstream gradient. We just use the chain rule. 


## Extra reading
- [CS231](https://www.youtube.com/watch?v=vT1JzLTH4G4)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- Kevin Clarke matrix calculus