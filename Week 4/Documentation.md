<script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Neural Network

## Model Representaion I

* Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). In our model, our dendrites are like the input features $ x_1\cdots x_n $, and the output is the result of our hypothesis function. In this model our $ x_0 $ input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, $ \frac{1}{1 + e^{-\theta^Tx}} $, yet we sometimes call it a sigmoid (logistic) activation function. In this situation, our "theta" parameters are sometimes called "weights". 

* Visually, a simplistic representation looks like:
$$
\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline \end{bmatrix}\rightarrow\begin{bmatrix}\ \ \ \newline \end{bmatrix}\rightarrow h_\theta(x)
$$

* Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".  <br/><br/>
We can have intermediate layers of nodes between the input and output layers called the "hidden layers."  <br/><br/>
In this example, we label these intermediate or "hidden" layer nodes $ a^2_0 \cdots a^2_n $ and call them "activation units."
$$
\begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \newline& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*}
$$

* If we had one hidden layer, it would look like:
$$
\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \newline a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x)
$$

* The values for each of the "activation" nodes is obtained as follows:
$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$

* This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $ \Theta^{(2)} $ containing the weights for our second layer of nodes.  <br/><br/>
Each layer gets its own matrix of weights, $ \Theta^{(j)} $.

* The dimensions of these matrices of weights is determined as follows:  
**If network has s<sub>j</sub> units in layer j and s<sub>j+1</sub> units in layer j+1, then Θ<sub>(j)</sub> will be of dimension s<sub>j+1</sub>×(s<sub>j</sub>+1).**

* The +1 comes from the addition in $ \Theta^{(j)} $ of the "bias nodes," $ x_0 $ and $ \Theta_0^{(j)} $. In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:  <br/><br/>
![](/Assets/Neural%20Network%20Model%20Representation%201.png)  

## Model Representaion II

* To re-iterate, the following is an example of a neural network:  
$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$

* In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable $ z_k^{(j)} $ that encompasses the parameters inside our g function. In our previous example if we replaced by the variable z for all the parameters we would get: 
$$
\begin{align*}a_1^{(2)} = g(z_1^{(2)}) \newline a_2^{(2)} = g(z_2^{(2)}) \newline a_3^{(2)} = g(z_3^{(2)}) \newline \end{align*}
$$

* In other words, for layer j=2 and node k, the variable z will be:
$$
\begin{align*}
z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n
\end{align*}
$$

* The vector representation of x and $ z^{j} $ is:
$$
\begin{align*}x = \begin{bmatrix}x_0 \newline x_1 \newline\cdots \newline x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \newline z_2^{(j)} \newline\cdots \newline z_n^{(j)}\end{bmatrix}\end{align*}
$$

* Setting $ x = a^{(1)} $ , we can rewrite the equation as:
$$
\begin{align*}
z^{(j)} = \Theta^{(j-1)}a^{(j-1)}
\end{align*}
$$

* We are multiplying our matrix $ \Theta^{(j-1)} $ with dimensions $ s_j \times (n+1) $ (where $ s_j $ is the number of our activation nodes) by our vector $ a^{(j-1)} $ with height (n+1). This gives us our vector $ z^{(j)} $ with height $ s_j $. Now we can get a vector of our activation nodes for layer j as follows:
$$
\begin{align*}
a^{(j)} = g(z^{(j)})
\end{align*}
$$

* Where our function g can be applied element-wise to our vector $ z^{(j)} $ .  <br/><br/>
We can then add a bias unit (equal to 1) to layer j after we have computed  $ a^{(j)} $. This will be element $ a_0^{(j)} $ and will be equal to 1. To compute our final hypothesis, let's first compute another z vector:
$$
z^{(j+1)} = \Theta^{(j)}a^{(j)}
$$  
* We get this final z vector by multiplying the next theta matrix after $ \Theta^{(j-1)} $ with the values of all the activation nodes we just got. This last theta matrix $ \Theta^{(j)} $ will have only one row which is multiplied by one column $ a^{(j)} $ so that our result is a single number. We then get our final result with:
$$
h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})
$$
* Notice that in this last step, between layer j and layer j+1, we are doing exactly the same thing as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

## Examples and Intuitions I
* A simple example of applying neural networks is by predicting $ x_1 $ AND $ x_2 $ , which is the logical 'and' operator and is only true if both $ x_1 $ and $ x_2 $ are 1.  <br/><br/>
The graph of our functions will look like:  
$$
\begin{align*}
    \begin{bmatrix}
        x_0 \newline
        x_1 \newline
        x_2
    \end{bmatrix}
    \rightarrow
    \begin{bmatrix}
        g(z^{(2)})
    \end{bmatrix}
    \rightarrow
    h_\Theta(x)
\end{align*}
$$

* Remember that $ x_0 $ is our bias variable and is always 1.  <br/><br/>
Let's set our first theta matrix as:
$$
\begin{align*}
    \Theta^{(1)} = 
    \begin{bmatrix}
        -30 & 20 & 20
    \end{bmatrix}
\end{align*}
$$

* This will cause the output of our hypothesis to only be positive if both $ x_1 $ and $ x_2 $ are 1. In other words:
$$
\begin{align*}& h_\Theta(x) = g(-30 + 20x_1 + 20x_2) \newline \newline & x_1 = 0 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-30) \approx 0 \newline & x_1 = 0 \ \ and \ \ x_2 = 1 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 1 \ \ then \ \ g(10) \approx 1\end{align*}
$$

* So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all the other logical gates. The following is an example of the logical operator 'OR', meaning either $ x_1 $ is true or $ x_2 $ is true, or both:  <br/><br/>
![](/Assets/Neural%20Network%20Example%201%20Figure%201.png)

* Where g(z) is the following:  <br/><br/>
![](/Assets/Neural%20Network%20Example%201%20Figure%202.png)

## Examples and Intuitions II

* The $\Theta^{(1)} $ matrices for AND, NOR, and OR are:
$$
\begin{align*}
    AND:\newline
    \Theta^{(1)} &=
    \begin{bmatrix}
        -30 & 20 & 20
    \end{bmatrix} \newline 
    NOR:\newline
    \Theta^{(1)} &= 
    \begin{bmatrix}
        10 & -20 & -20
    \end{bmatrix} \newline 
    OR:\newline
    \Theta^{(1)} &= 
    \begin{bmatrix}
        -10 & 20 & 20
    \end{bmatrix} \newline
\end{align*}
$$

* We can combine these to get the XNOR logical operator (which gives 1 if $ x_1 $ and $ x_2 $ are both 0 or both 1).
$$
\begin{align*}
    \begin{bmatrix}
        x_0 \newline x_1 \newline x_2
    \end{bmatrix} 
    \rightarrow
    \begin{bmatrix}
        a_1^{(2)} \newline a_2^{(2)} 
    \end{bmatrix} 
    \rightarrow
    \begin{bmatrix}
        a^{(3)}
    \end{bmatrix} 
    \rightarrow 
    h_\Theta(x)
\end{align*}
$$

* For the transition between the first and second layer, we'll use a $ \Theta^{(1)} $ matrix that combines the values for AND and NOR:
$$
\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20 \newline 10 & -20 & -20\end{bmatrix}
$$

* For the transition between the second and third layer, we'll use a $ Θ^{(2)} $ matrix that uses the value for OR:
$$
\Theta^{(2)} =\begin{bmatrix}-10 & 20 & 20\end{bmatrix}
$$

* Let's write out the values for all our nodes:
$$
\begin{align*}& a^{(2)} = g(\Theta^{(1)} \cdot x) \newline& a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) \newline& h_\Theta(x) = a^{(3)}\end{align*}
$$

* And there we have the XNOR operator using a hidden layer with two nodes! The following summarizes the above algorithm:  <br/><br/>
![](/Assets/Neural%20Network%20Example%202%20Figure%201.png)

## Multiclass Classification

* To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four categories. We will use the following example to see how this classification is done. This algorithm takes as input an image and classifies it accordingly:  <br/><br/>
![](/Assets/Neural%20Network%20Multiclass%20Classification%201.png)

* We can define our set of resulting classes as y:  <br/><br/>
![](/Assets/Neural%20Network%20Multiclass%20Classification%202.png)

* Each $ y^{(i)} $ represents a different image corresponding to either a car, pedestrian, truck, or motorcycle. The inner layers, each provide us with some new information which leads to our final hypothesis function. The setup looks like:  <br/><br/>
![](/Assets/Neural%20Network%20Multiclass%20Classification%203.png)

* Our resulting hypothesis for one set of inputs may look like:
$$
h_\Theta(x) =\begin{bmatrix}0 \newline 0 \newline 1 \newline 0 \newline\end{bmatrix}
$$

* In which case our resulting class is the third one down, or $ h_\Theta(x)_3 $ , which represents the motorcycle. 

# Lecture Notes:
1. [Neural Network and Non-linear hyotheses](/Week%204/Lecture%20Notes/Lecture%208.pdf)