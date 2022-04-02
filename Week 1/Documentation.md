# Welcome to Machine Learning!

* We are using machine learning every day without knowing it. Example:
    * Google and Bing search uses machine learning to rank web pages.
    * Facebook and Apple's application also include machine learning.
    * Email spam detection is another example.

# What is Machine Learning?

* **Arthur Samuel** described it as: *"the field of study that gives computers the ability to learn without being explicitly programmed."*  
This is an older, informal definition.

* **Tom Mitchell** provides a more modern definition: *"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."*  

    Example: playing checkers.  
    E = the experience of playing many games of checkers  
    T = the task of playing checkers.  
    P = the probability that the program will win the next game.

* In general, any machine learning problem can be assigned to one of two broad classifications:  
    1. Supervised Learning
    2. Unsupervised Learning

## Supervised Learning
---

* In **supervised learning**, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

* **Supervised learning problems** are categorized into:  
    1. Regression
    2. Classification

* In a **regression problem**, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. 

* In a **classification problem**, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

* Examples 1:  
    * Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.  

    * We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

* Example 2:
    * Regression - Given a picture of a person, we have to predict their age on the basis of the given picture

    * Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 

## Unsupervised Learning

---

* **Unsupervised learning** allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

* We can derive this structure by clustering the data based on relationships among the variables in the data.

* With unsupervised learning there is no feedback based on the prediction results.

* Example:
    
    1. **Clustering**: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

    2. **Non-clustering**: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

# Model Representation

* To establish notation for future use, we’ll use x<sup>(i)</sup> to denote the ***input*** variables (living area in this example), also called ***input features***, and y<sup>(i)</sup> to denote the ***output*** or ***target*** variable that we are trying to predict (price). A pair (x<sup>(i)</sup> , y<sup>(i)</sup>) is called a **training example**, and the dataset that we’ll be using to learn - a list of m training examples (x<sup>(i)</sup> , y<sup>(i)</sup>); i = 1, . . . , m - is called a **training set**.  

* Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values. In this example, X = Y = ℝ. 

* To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a ***good*** predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.

* Seen pictorially, the process is therefore like this:  
![Model Representation Image](/Assets/Model%20Representation%201.png "Model Representation" )  

* When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. 

* When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

# Cost Function
* We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.  
![Squared error function](/Assets/Cost%20Function%201.png "Squared Error Function")  
To break it apart, it is  <sup>1</sup>/<sub>2</sub> x&#772;&#772; where x&#772;&#772; is the mean of the squares of h<sub>θ</sub>(x<sub>i</sub>)−y<sub>i</sub>, or the difference between the predicted value and the actual value.

* his function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved (<sup>1</sup>/<sub>2</sub>) as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the <sup>1</sup>/<sub>2</sub> term.  

## Cost Function Intuition I
---

* If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line h<sub>θ</sub> (x) which passes through these scattered data points.  

* Our objective is to get the **best possible line**. The best possible line will be such so that ***the average squared vertical distances of the scattered points from the line will be the least***. Ideally, the line should pass through all the points of our training data set. In such a case, the value of J (θ<sub>0</sub>, θ<sub>1</sub>) will be 0.  
<img src="../Assets/Cost Function 2.png" alt="h theta of x" width="300"/>
<img src="../Assets/Cost%20Function%203.png" alt="J of theta 1" width="300"/>

* When θ<sub>1</sub> = 1, we get a slope of 1 which goes through every single data point in our model. Conversely, when θ<sub>1</sub> = 0.5, we see the vertical distance from our fit to the data points increase. This increases our cost function to 0.58.  
<img src="../Assets/Cost%20Function%204.png" alt="h_theta of x" width="300"/>
<img src="../Assets/Cost%20Function%205.png" alt="h_theta of x" width="300"/>

* Thus as a goal, ***we should try to minimize the cost function***. In this case, θ<sub>1</sub> = 1 is our global minimum.  
<img src="../Assets/Cost%20Function%206.png" alt="h_theta of x" width="300"/>

## Cost Function Intuition 2
---

* A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.  
<img src="../Assets/Cost%20Function%207.png" alt="h_theta of x" width="500"/>  

* Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for J(θ<sub>0</sub>, θ<sub>1</sub>) and as a result, they are found along the same line.  
The circled x displays the value of the cost function for the graph on the left when θ<sub>0</sub> = 800 and θ<sub>1</sub> = -0.15.  
Taking another h(x) and plotting its contour plot, one gets the following graphs:  
<img src="../Assets/Cost%20Function%208.png" alt="h_theta of x" width="500"/>  

* When θ<sub>0</sub> = 360 and θ<sub>1</sub> = 0, the value of J(θ<sub>0</sub>, θ<sub>1</sub>) in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

* The graph above minimizes the cost function as much as possible and consequently, the result of θ<sub>0</sub> and θ<sub>1</sub> tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

# Gradient Descent

* So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

* Imagine that we graph our hypothesis function based on its fields θ<sub>0</sub> and θ<sub>1</sub> (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

* We put θ<sub>0</sub> on the x axis and θ<sub>1</sub> on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.  
<img src="../Assets/Gradient%20Descent.png" alt="h_theta of x" width="500"/>  

* We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum.  The red arrows show the minimum points in the graph.

* The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate. 

* For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of J(θ<sub>0</sub>, θ<sub>1</sub>). Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places. 

* The gradient descent algorithm is:  
repeat until convergence:  
***θ<sub>j</sub> := θ<sub>j</sub> − α <sup>∂</sup>/<sub>∂θ</sub> J(θ<sub>0</sub>, θ<sub>1</sub>)***  
where j=0, 1 represents the feature index number.

* At each iteration j, one should simultaneously update the parameters ***θ<sub>1</sub>, θ<sub>2</sub>,...,θ<sub>n</sub>***. Updating a specific parameter prior to calculating another one on the ***j<sup>(th)</sup>*** iteration would yield to a wrong implementation. 

## Gradient Descent Intuition
---

* In this video we explored the scenario where we used one parameter θ<sup>1</sup> and plotted its cost function to implement a gradient descent. Our formula for a single parameter was :  
Repeat until convergence:  
***θ<sub>1</sub> := θ<sub>1</sub> − α <sup>d</sup>/<sub>dθ<sub>1</sub></sub> J(θ<sub>1</sub>)***

* Regardless of the slope's sign for ***d</sup>/<sub>dθ<sub>1</sub></sub> J(θ<sub>1</sub>)***, ***θ<sub>1</sub>*** eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of ***θ<sub>1</sub>*** increases and when it is positive, the value of ***θ<sub>1</sub>*** decreases.

* On a side note, we should adjust our parameter ***α*** to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong. <br/>  
* **How does gradient descent converge with a fixed step size** ***α*** **?** <br/>  
The intuition behind the convergence is that ***d</sup>/<sub>dθ<sub>1</sub></sub> J(θ<sub>1</sub>)*** approaches 0 as we approach the bottom of our convex function. At the minimum, the derivative will always be 0 and thus we get:  
***θ<sub>1</sub> := θ<sub>1</sub> − α * 0***

# Gradient Descent For Linear Regression

* When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :  
<img src="../Assets/Gradient%20Descent%20For%20Linear%20Regression%201.png" alt="Gradient Descent For Linear Regression" width="500"/>  
where m is the size of the training set, θ<sub>0</sub> a constant that will be changing simultaneously with θ<sub>1</sub> and x<sub>i</sub> ,y<sub>i</sub> are values of the given training set (data).  
Note that we have separated out the two cases for θ<sub>j</sub> into separate equations for θ<sub>0</sub> and θ<sub>1</sub>; and that for θ<sub>1</sub>we are multiplying x<sub>i</sub> at the end due to the derivative.  
The following is a derivation of ***<sup>∂</sup>/<sub>∂θ</sub> J (θ)*** for a single example:  
<img src="../Assets/Gradient%20Descent%20For%20Linear%20Regression%202.png" alt="Gradient Descent For Linear Regression" width="300"/>  

* The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

* So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**.  
Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.  
<img src="../Assets/Gradient%20Descent%20For%20Linear%20Regression%202.png" alt="Gradient Descent For Linear Regression" width="300"/> 

* The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.

# Matrices and Vectors

* **Matrices** are 2-dimensional arrays. 
* A **vector** is a matrix with one column and many rows.  
**Notation and terms:**
    * ***A<sub>ij</sub>*** refers to the element in the ith row and jth column of matrix A.
    * A vector with 'n' rows is referred to as an 'n'-dimensional vector.
    * ***v<sub>i</sub>*** refers to the element in the ith row of the vector.
    * In general, all our vectors and matrices will be 1-indexed.  
    Note that for some programming languages, the arrays are 0-indexed.
    * Matrices are usually denoted by uppercase names while vectors are lowercase.
    * "Scalar" means that an object is a single value, not a vector or matrix.
    * ℝ refers to the set of scalar real numbers.
    * ℝ<sup>n</sup> refers to the set of n-dimensional vectors of real numbers.

## Addition and Scalar Multiplication
---

* Addition and subtraction are element-wise, so you simply add or subtract each corresponding element:  <br/><br/>
![Matrix Addition](/Assets/Matrix%20Addition%201.png "Matrix Addition")  
Subtracting Matrices:  <br/><br/>
![Matrix Subtraction](/Assets/Matrix%20Addition%202.png "Matrix Subtraction")  
To add or subtract two matrices, their dimensions must be the same.

* In scalar multiplication, we simply multiply every element by the scalar value:  <br/><br/>
![Matrix Scalar Multiplication](/Assets/Matrix%20Addition%203.png "Matrix Scalar Multiplication")  
In scalar division, we simply divide every element by the scalar value:  <br/><br/>
![Matrix Scalar Division](/Assets/Matrix%20Addition%204.png "Matrix Scalar Division")  

## Matrix-Vector Multiplication
---

* We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.  <br/><br/>
![Matrix-Vector Multiplication](/Assets/Matrix-Vector%20Multiplication%201.png "Matrix-Vector Multiplication")  
The result is a vector. The number of columns of the matrix must equal the number of rows of the vector.

* An m x n matrix multiplied by an n x 1 vector results in an m x 1 vector.

## Matrix-Matrix Multiplication
---

* We multiply two matrices by breaking it into several vector multiplications and concatenating the result.  <br/><br/>
![Matrix-Matrix Multiplication](/Assets/Matrix-Matrix%20Multiplication%201.png "Matrix-Matrix Multiplication")  
An m x n matrix multiplied by an n x o matrix results in an m x o matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

* To multiply two matrices, the number of columns of the first matrix must equal the number of rows of the second matrix.

### Matrix Multiplication Properties:
* Matrices are not commutative: A∗B &#8800; B∗A  
* Matrices are associative: (A∗B)∗C =  A∗(B∗C)(A∗B)∗C=A∗(B∗C)  

* The identity matrix, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.  <br/><br/>
![Matrix-Matrix Multiplication](/Assets/Identity%20Matrix.png "Matrix-Matrix Multiplication")
* When multiplying the identity matrix after some matrix (A∗I), the square identity matrix's dimension should match the other matrix's columns. When multiplying the identity matrix before some other matrix (I∗A), the square identity matrix's dimension should match the other matrix's rows.

## Inverse and Transpose
---

* The inverse of a matrix A is denoted A<sup>−1</sup>. Multiplying by the inverse results in the identity matrix.

* A non square matrix does not have an inverse matrix. Matrices that don't have an inverse are singular or degenerate.

* The transposition of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it.  <br><br>
![Matrix Transpose](/Assets/Matrix%20Transpose.png "Matrix Transpose")  

# Lecture Notes:
1. [Introduction to Machine Learning](/Week%201/Lecture%20Notes/Lecture%201.pdf)
2. [Model Representation](/Week%201/Lecture%20Notes/Lecture%202.pdf)
3. [Matrices and Vectors](/Week%201/Lecture%20Notes/Lecture%203.pdf)