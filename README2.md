# PART- II Post preprocessing

**Here, we create the architecture of neural network, train, and test he model.** 

## Creating the architecture of the Neural Network

* Initialization of the RBM class

    **RBM IS A probabilistic graphical model.**
    We will create (RBM) class with 4 functions:
    
    1- Initialize the RBM object
    
    2- Sample h that will sample the probabilities of the hidden nodes given the visible nodes
    
    3- Sample v that will sample the probabilities of the visible nodes given the hidden nodes
    
    4- Training the model

    * __ init __ function creates parameters of the object that will be created
    
        *Self refers to the obj that will be created afterwards.*
        
        > Nv-no. Of visible nodes
        
        > Nb-no. Hidden nodes

        Attributes w,a,b are initialized

        * w=Initialize weights for probabilities of size nh,nv according to normal distribution.
        * a=Probability for bias For the probability of the hidden nodes when visible nodes are given. 
        * b=Probability for bias For the probability of the visible nodes when hidden nodes are given. 
        *(1,nv/nb) 1 is here to add a dimension and to make it a 2D tensor.* 

    * 2nd function sample_h
        
        **returns p_h_given_v and some samples of hidden nodes of our rbm.**

        > H- hidden nodes
        
        > V- visible node/neurons (input vectors of our observation with all the ratings)
        
        > P(prob) of h(hidden nodes) given v(visible nodes)(p_h_given _v)- activation fn -The probability of the given (h)hidden node is 1 given value of v(visible node)
 
        *(p_h_given_v)Probability of hidden nodes given visible nodes, once we have the probability we can sample the activations of the hidden nodes returns some samples of hidden nodes.*

        * X (parameter for sample_h function)- visible neurons v in the probabilities p_h_given_v
        * torch.mm- product of two torch tensors here, x(visble neurons),w(tensor of weights), t()-transpose 
        * activation- Sigmoid activation fn is applied to w(weight vector) product x(vsivble neurons vector) +bias. Now dealing with hidden nodes so take bias of hidden nodes (a).
        * Activation variable will be inside the activation fn.
           Here each input Vector will not be treated individually But inside bathces\that is the new dimension we created while making biases  a,b - minibathces.
           While adding bias to the activation variable we will make sure that it is added to every Mini batch, so we add a dimension to the biases we are adding -expand_as(as what we want to expand the bias) here wx. 
        * p_h_given_v= Torch.sigmoid-sigmoid activation fn- is the probability that the hidden node Will be activated according to the value of the visible load.
        * In the final step - return samples of all the hidden nodes according to p_h_given v(ector of nh elements), each of these elements of the vector corresponds to each of the nh nodes, Each of these elements is the Probability that the hidden node is activated 
 
              * Any element in the p_h_given_v vector Is the probability that the hidden node is activated But remember that given the values of the visible nodes 
              * If a person likes drama movies so his is visible nodes will be 1 therefore p_h_given_v for drama movies will be high.
              * To sample for each of the hidden nodes while depending on the probabilities for these hidden nodes in p_h_given_v . We will activate yes or no this hidden neuron, 
              * Suppose for the hidden neuron the probability corresponding to that in a neuron In this ph given v is >0.7 we will not activate the neuron and <0.7 will activate. that’s how Bernoulli sampling works.
              * 0 response to the hidden noDes which were not activated after the sampling and 1 for activated.


* Sample _h first function for gibbs sampling other is sample _v

We have sampled the hidden nodes according to the probabilities p_h_given_v

3rd function sample_v

Pv given h Given the values of hidden nodes , we ret the pro of that ec of the visible nodes=1, then return some samples based on bernoulli sampling

1682 movies - a vector of 1682 visible nodes -1682 probabilities that the visible neuron =1 given the activations of the hidden nodes.

There x represented visible nodes, y represent the values of the hidden nodes. And also use hidden node bias, we will not take transpose becoz w is the weight matrix for p v given h

Train fn to take care of contrastive divergence 
Rbm -Energy-based model We're trying to minimise the energy function

Also, a probabilistic graphical model where we maximize the log-likelihood of the training set

 to maximize the log-likelihood or to minimise the energy function we need to calculate the gradient. Direct computations are too heavy so we will approximate the gradients with the help of contrastive divergence.

Gibbs sampling consists of creating this Gibbs chain in k-steps and this Gibb chain in created exactly by sampling K times the hidden nodes and the visible node.
That is that, you know, we start with our input vector, V0, then based on the probabilities PH given V zero , we sample the first set of nodes so that's at the first iteration.
Then we take these sampled hidden nodes as input. Let's call them H one to sample the visible nodes with the probabilities PV given H one and then again we use these sample visible nodes.
Let's call them V one, to sample again the hidden nodes with the probabilities PH given V one and then again we sample the visible nodes and we sample the hidden nodes and we do this K times. 
And that's exactly what this CDK algorithm is about.

Arguments-
V0- input vector consists of the user’s ratings
Vk- visible nodes after obtaining the rating of all movies of one user, later we will make a loop of all users obtained after k samplings.
Ph0- vector of probabilities that at the 1st iteration the hidden nodes= 1 given the v0 (input vector of observations). 
Phk- probabilities of the hidden nodes equal to 1 after k sampling given the values of visible nodes vk

Now update w (weight),a(visible node bias) ,b(hidden node bias)

w=w(weight)+product(torch.mm does it)(ph0,v0) - product(vk,phk)
b=b+torch.sum((v0-vk)),0) 0 added to maintain b in its original format i.e 2d tensor
a=a+torch.sum((ph0-phk),0)

Rbm class finishes here, now we will create an object of rbm clas and then train it over several epochs To find the Optimal weights that will allow us to predict the ratings of the objects that were not originally rated.

Nv-no. Of visible nodes is 1682 i.e nb_movies but we can also take is as len(training_set[0])

Nh-we can choose any no.- number of features we want to detect like oscar, actor yr actors in it (tunable for better results)

Batch_size- weights not updated after each observation but updated after several observations of a single batch (tunable for better results)

Now we create an object rbm of the RBM class.





## Training the RBM

Nb_epoch - no. Of epochs
We have 943 observations i.e rows in the training set

train_loss= loss function to measure the error between the real ratings and the predictions

We need to normalize the loss fn so we need to divide it by a counter, so initialize a counter i.e,  s

We have to loop over all the users in batches too as the functions in the RBM class are meant for a single user.

range(0,843,100)-0-99,100-101 we want

**Vk- input batch (id_user:id_user+batch_size)-all the users from id_user to next 100 in the training set -output after Gibbs sampling, i.e, after k steps of random walk.

**V0-(target) batch of original ratings that’s not gonna change and be compared with predicted values at last.
 In the beginning, the input is the same as the target

**Ph0- to get this we will use sample_h method which we have created earlier (ph0,_)-which means we just want the first element of the output not to include the Bernoulli samples.

X-visible nodes, that are at the start i.e v0

Another for loop for k steps of contrastive divergence

Gibbs sampling consists of making the Gibbs chain of several Round trips from the visible note to the hidden note Then from the hidden note the visible nodes’. In each round trip of this Gibbs chain of Gibbs sampling, the visible nodes are updated.

Call sample_h method on the visible node(vk becoz v0 is our target we don’t want to change and vk will be updated later) to get the first sampled hidden nodes
 
Vk is updated by sample_v  function which will give sampled visible nodes

At the end of the loop, we will get the 10th sample of hidden and visible nodes 

Now we can approximate the gradients

We don't wanna learn where the user haven’t rated movies i.e nodes as -1
So vk[v0<0]< ensures -1 in vk is kept same as in v0 i,.e -1

We will apply train fn to update the weights and the biases

We don’t have phk variable, so we will compute phk, phk,_ we use the sample_ h fn on the last sample of the visible nodes after the 10 steps i.e vk

Now we will train the RBM by using the train method

Now we measure the loss by updating train_loss by adding the error. we will jst use rmse in autoencoder here we will use the simple distance between our target(v0) and our prediction(vk)


[v0>=0] - to use only existing ratings in the training i.e exclude the movies which were not rated.

Update the counter which normalizes the training loss
Print-to see what is happening

In each epoch various minibatches are trained . epoch is basically training model again and again.

Testing the RBM on testset

Not use epochs, the batch size they were specific to training only so remove all terms related to those

Vt- target, original ratings of the user in the test set
V- 

  get the output.

Right now the training set contains the ratings of the training set and it doesn't contain the answers of the test set.

But, by using the inputs of the training set we will activate the neurons of our RBM to predict the ratings of the movies that were not rated yet, and that is the ratings of the test set.

So we need this as input to get the predicted ratings of the test set.
Because we are getting these predicted ratings from the inputs of the training set
that are used to activate the neurons of our RBM.

we are using the inputs of the training set to activate the neurons of the RBM to get the predicted ratings of the test set.

We need to take one step rather than 10 steps

Rating 

 an if condition
because you know that's always the same idea we want to test
the predictions to the ratings of the test set
that actually exist.
You know, in the test set we still have some ratings
that are existent in the test set
but we also still have some minus one ratings.
So the minus one are ratings that just never happened.
Whether it was in the training set or in the test set.
And we don't want to consider these ratings in the test set,
of course, that's always the same id.


 so we are making this if condition
to filter these non-existent ratings of the test set.
vk[v0<0] = v0[v0<0
We remove it becoz no training so  longer reqiured.
 
 
phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)

Now as we are not doing training so no need to update the weights so let’s get rid of them

vt>=0, we are getting the indexes that have existent ratings


V0- target isnt changed
Vk-fianl input after k samplng

Out of 4, we are predicting 3 correctly.
