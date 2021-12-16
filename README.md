# Back propagation Neural Networks
 Implementing the back propagation algorithm and  feed forward

In both , we read the input (x vector) and
the output (y vector) from a file. Below is the structure of the input
file:
1) First line: M, L, N where M is number of Input Nodes, L is
number of Hidden Nodes and N is number of Output Nodes
2) Second line: K, the number of training examples, each line has
length M+N values, first M values are X vector and last N values
are output values.
3) K lines follow as described
An example of input file (just for clarification not to be used):
3 2 2
3
1 1 1.5 2 2
-1 2.25 0.5 -0.5 1.2
1 1 1 1 2

For back propagation:
After reading the input file, we do a normalization step
on the input features.
Normalization is done by computing, for each numeric
x-data column value v, v' = (v - mean) / std dev. 
We perform back propagation algorithm, The algorithm should stop after running for 500 iterations,After that we print MSE on the screen and save final weights to a file.

For feed forward:
perform feed forward algorithm on the input data using the best weights we have calculated in back propagation. Print the MSE.