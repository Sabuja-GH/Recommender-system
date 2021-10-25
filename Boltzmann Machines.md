# Boltzmann Machines

    Nodes- hidden, visible , all neurons/nodes are interconnected. No output layer.
<img align="Center" height="330px"  src="https://user-images.githubusercontent.com/85345738/138544834-f4995292-4551-400f-8be1-adf0408a8f91.png" />

Higher the energy of any state then lower the probability of our system to be in that state, Boltzmann distribution formula.

(EBM)energy-based model weight adjusted so that the system remains at its lowest energy state as per Boltzman equation. 

Full Boltzmann machine is hard to implement because so many nodes and so many connections make it hard, so instead we use **Restricted Boltzmann machines**(RBMs).

In RBM hidden and visible nodes are not connected to each other
<img align="Center" height="250px"  src="https://user-images.githubusercontent.com/85345738/138545055-cfc0615c-158c-45e7-a048-fe1e235f9544.png" />

Movie recommender system- Our model is specific to single sysem rather than a model for other systems or in general. 

From the inputs (visible) nodes of our model the hidden nodes in the RBM will find out features that have a higher say in the person's movie taste( like genre, Oscars, director etc) and so the weights are adjusted accordingly.

RBM is non-directional model unlike the previous NN models. Unlike ANN here we can’t forward propagate and backpropagate and adjust weights with the help of gradient descent.

Here, we will use **contrastive divergence** to bring our system to a minimum energy state by using Gibbs sampling process which also gives the final inputs (our required output).

* Initially, we use randomly assigned weights we find the hidden nodes and then by using the exact same weights we gonna reconstruct the input(visible nodes).

* The reconstructed inputs will not be the same as the initial input nodes even though We are using the exact same weights.  *This is because the hidden nodes are not created from a single input node rather a single hidden node is the result of the interaction of all input nodes.* 

* We will use the new input nodes and reconstruct the next hidden node and this process of creating input and hidden nodes goes on and on until an input node exactly match with the previous input node. 

* The weights are fixed throughout the process. 

At last, we are trying to adjust the energy curve (energy vs dataspace) by modifying the weights in order to facilitate/create a system that best resembles our input values.

(((Initially, some values of the input node will be null i,e, the movies which the user haven’t seen. At last, those input nodes will be having some values.)))

<img align="Center" height="250px"  src="https://user-images.githubusercontent.com/85345738/138544385-c8faad0a-7010-4a83-ad22-e65fdbe575d1.png" />

RBMs stacked over one another - Deep belief networks(DBN).

<img align="Center" height="250px"  src="https://user-images.githubusercontent.com/85345738/138544379-c0ad2da0-6072-40b2-9a72-7c1dee5f28cc.png" />

Deep Boltzmann machine not equal deep belief networks(extract more complex features).
