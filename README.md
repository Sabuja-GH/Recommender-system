# Recommender-system
A  restricted Boltzmann machine (RBM) movie recommender system.

In this recommender system we will find out if the customer is gonna like the movie - yes or no (0 or 1).

In the next recommender system i.e, autoencoders we will create a system that gives the rating of the movie user likely to give (1-5 stars)

##  Preprocessing part
**RBM- Restricted Boltzmann machines**

* Importing the libraries
    * Numpy-to work with arrays
    * Pandas- to import the dataset, crate training and testset
    * Nn-module of torch to implement neural network
    * Parallel- for parallel computation
    * Optim- for an optimizer

* Importing Movies dataset
    * Sep =’::’ cant use default, as it is a comma(,) CSV(comma separated file) becoz a movie title may have a comma so the same movie will be divided into two columns. So we are using :: as to separate values.
    * The data has no header (names of columns), header =None , default was ‘infer’.
    * Engine- used to ensure the dataset is imported correctly, we using ‘python’ engine to make it efficient.
    * Encoding- we need to use a different  encoding than usual because some of the title has special characters that, default was UTF-8 now we will use ‘latin-1’.

Users , rating dataset will be importes in the same manner as above.

* Prepare training and test set
    > u1.base=training

    > u1.test=test set

    * Here the separator is a tab(‘\t’) not ‘::’ not default i.e ‘,’ too. 
    * Now change the training dataset into array by using np.array.
    * Now, we will do the for testset too.

    *Many test and base tests, so we can do k fold cross-validation.*

* We need to find the total number of users and movies by finding the max of movies and users id that may be present in test or training set.

* Converting the data into an array with users in lines and movies in columns (matrices)

    *We wish to create 2 matrices (test, train) where each column is a movie and each row represents the user from the training and test set that we imported previously, if a user hasn't rated a movie we put 0.*
    
    *Now convert our data into array(matrix) where users are in lines and movies in columns.*
    * To do this we will make a function named convert. 
    * We will create a list of lists because torch expects inputs as such.i.e, new_data, one list for each user, 943 lists as 943 users, each of this list will have 10682 elements because of 10682 movies.
    * Id_movies has movies that the user have rated.
    * Id_rating has the ratings for movies that the user have rated.
    * Ratings[id_movies-1] bcoz indexing in python starts with 0.

* Converting data into Torch tensors

    *Tensors are simply arrays whose each element are of a single data type,multi-dimensional array but instead of being NumPy array, they are Pytorch array.*

    *We could have used np arrays but that would be less efficient. There are also TensorFlow tensors we could have used instead of PyTorch tensors but implantation and final results are better in PyTorch tensors for our recommender system.*

    * Torch.FloatTensor creates a pytrch tensor which is a matrix that consists of a single datatype float, takes input as a list of lists.

From now on all the implementation will be specific for the Boltzmann machines model which will predict if a person/user will like a movie or not

* Convert ratings into 0 and 1 (liked or disliked) for both training and test set, 
And 0 to -1(movies user haven’t seen) 

    * ‘Or’ operator doesn’t work in PyTorch so we are using multiple lines to covert ratings to binary values(0/1)
