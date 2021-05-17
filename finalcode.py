#Applying PCA to Optdigits dataset and selecting features that explain 90% of the variance and Clustering with Kmeans and reporting accuracy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random as rd

optdigits_df=pd.read_csv('optdigits_csv.csv') #Loading the dataset from the csv file

data_features=optdigits_df.iloc[:,:64] #Here we are selecing the first 64 columns which contains our data

data_labels=optdigits_df.iloc[:,64:] #Here we are selecing the last column which contains the labels associated with our data

firstimage=data_features.iloc[0,:].to_numpy().reshape(8,8) #Here we are taking the first row and reshaping it into the size of the original image and then visualising it
plt.imshow(firstimage) #This function visualises the image
print("Label of this first image : ",int(data_labels.iloc[0])) #Here we are printing the label associated with the first image

b=optdigits_df.describe() #Here we are describing the dataset and hence using it's max value to plot our bar graph that helps visualise the range of values that our dataset covers
trans=b.T
fig=plt.figure(figsize=(20,10))
plt.xlabel('Inputs')
plt.ylabel('Maximum')
plt.title('Maximum of Inputs')
print(trans.describe())
trans['max'].plot(kind='bar')

b=optdigits_df.describe() #Here we are describing the dataset and then taking the mean of the columns and plotting it in a bar graph. This helps visualise the mean values across all the columns in our dataset
trans=b.T
fig=plt.figure(figsize=(20,10))
plt.xlabel('Inputs')
plt.ylabel('Mean')
plt.title('Mean of inputs')
print(trans.describe())
trans['mean'].plot(kind='bar')

def normalising_function(X):
    X_standard=StandardScaler().fit_transform(X)  #Scaling the dataset so that the range of values will be the same for all columns and thus helps in computation of values
    X_mean=np.mean(X_standard,axis=0)    #Finding the mean 
    X_cov=(X_standard - X_mean).T.dot((X_standard - X_mean))/(X_standard.shape[0]-1)  #Finding the covariance matrix
    return X_standard,X_mean,X_cov #returning the calculted matrices

def Principal_Component_Analysis(X):
    X_standard,X_mean,X_cov=normalising_function(X) #Storing the retunned matrices from the normalising_function()
    
    X_standard=X_standard-X_standard.mean(axis=0) #Subtracting the mean of each column from the dataset
    
    fig=plt.figure(figsize=(10,10))
    sns.heatmap(pd.DataFrame(X_cov))  #Creating a heatmap of the covariance matrix
    plt.show()
    
    eigenvalues, eigenvectors = np.linalg.eig(X_cov)  #Computing the eigen values and eigen vectors from the covariance matrix
    
    #The eigen vector corresponding to an eigen value is made into a pair and stored
    unsorted_eigen_value_vector_pair=[(np.abs(eigenvalues[i]),eigenvectors[:,i]) for i in range(len(eigenvalues))]
    #Then it is sorted in the descreasing order and we get out eigen value-vector pair
    sorted_eigen_value_vector_pair=sorted(unsorted_eigen_value_vector_pair, reverse=True, key=lambda x:x[0])

    #Plotting a heatmap of the eigen vector pairs
    #This helps in visualing which eigen vectors are more prominent
    #As the more prominent value in the heatmap is the one which has greater intensity
    fig=plt.figure(figsize=(15,4))
    sns.heatmap(pd.DataFrame([pair[1] for pair in sorted_eigen_value_vector_pair[0:21]]),annot=False,cmap='coolwarm',vmin=-0.5,vmax=0.5)
    plt.ylabel("Ranked Eigen Values")
    plt.xlabel("Eigen Vector Components")
    plt.show()
    
    #We are calculating the variance explained by each eigen value
    lam_sum=sum(eigenvalues)
    explained_variance=[(lam_k/lam_sum) for lam_k in sorted(eigenvalues,reverse=True)]
    
    #Cumulatively a certain number of features will help explain the entire dataset
    #Here we are creating a scree graph to show the explained variance ratio
    plt.figure(figsize=(6,4))
    plt.bar(range(len(explained_variance)),explained_variance,alpha=0.5,align='center',label='Individual Explained variance $\lambda_{k}$')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Ranked Principal Components')
    plt.title('Scree Graph')
    plt.legend(loc='best')
    plt.tight_layout()
    
    #This graph is cumulatively showing the variance explained by the features
    #As required in the question we can see from the graph that around 0.9 of variance is explained by around 30-35 of the features
    fig = plt.figure(figsize=(6,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(np.cumsum(explained_variance))
    ax1.set_ylim([0,1.0])
    ax1.set_xlabel('Number of Principal Components')
    ax1.set_ylabel('Cumulative explained variance')
    ax1.set_title('Explained Variance')
    plt.show()

    #We are printing the column indexes and the variance they explain
    #Once again this is a cumulative value
    #This is done to solidify our assumption that the principal components that explain 90% of variance are actually the first 32 eigen value-vector pairs
    print("\nCumulative variance explained by the features is shown here:")
    print([(j, np.cumsum(explained_variance)[j]) for j in range(len(explained_variance[:64]))])
    
    print("\nChoosing 5 Principal Components explains : ",np.cumsum(explained_variance)[5], "% of variance")
    print("Choosing 25 Principal Components explains : ",np.cumsum(explained_variance)[25], "% of variance")
    print("Choosing 32 Principal Components explains : ",np.cumsum(explained_variance)[32], "% of variance\n")

    
    #Finally we are creating our W.Transpose matrix of the most important components 
    #This will be multiplied with the dataset to reduce the features (because we are recasting or projecting the original dataset onto these principal components)
    matW = np.hstack( pair[1].reshape(64,1) for pair in sorted_eigen_value_vector_pair[0:32])#[0:4] originally
    print("\nShape of W.T matrix : ",matW.shape)

    #Multipling and returning the reduced features matrix
    Z = X_standard.dot(matW)
    print("Shape of returned features : ",Z.shape)
    
    return Z

reduced_data_features=Principal_Component_Analysis(data_features)

pca_reduced_dataset = pd.DataFrame(reduced_data_features) 

pca_reduced_dataset['label'] =data_labels

def test_train_splitter(reduced_data_features):
    train,test = train_test_split(reduced_data_features,test_size=0.20, random_state=1)
    division = train.shape[1] - 1
    train_without_label = train.iloc[:,0:division].values 
    train_with_label = train.iloc[:,:].values
    test_data = test.iloc[:,0:division].values  
    test_label = test.iloc[:,division].values 
    return train_with_label,train_without_label,test_data,test_label

def centroid_initialization(train_with_label,train_without_label):    
    i = 0
    Centroids=np.array([]).reshape(no_of_features,0)
    while(i<10):
        rand=rd.randint(0,m-1)
        # A point is guessed and label is correlated with i whose value varies from 0 - 10
        # If value of label is mached with i it is added in centroids
        # i is incremented
        # This step help us ensure that in cluster 0 centroid with feature whose label is 0 is assigned
        # This helps us as cluster zero most probably will point to digit 0
        if(train_with_label[rand,no_of_features] == i):
            Centroids=np.c_[Centroids,train_without_label[rand]]
            i = i + 1
    return Centroids

def kmeans_algorithm(train_with_label,train_without_label,Centroids):
    num_iter=100
    Output={} #output that will store each point in a cluster
    for n in range(100):
        EuclidianDistance=np.array([]).reshape(m,0)
        for k in range(K):
            tempDist=np.sum((train_without_label-Centroids[:,k])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C=np.argmin(EuclidianDistance,axis=1) # cluster for each point based on shortest distance from a centroid 
        #Calculating new mean
        Z={}
        for k in range(K):
            Z[k]=np.array([]).reshape(no_of_features,0)
            Output[k]=np.array([]).reshape(no_of_features+1,0)
        for i in range(m):
            Z[C[i]]=np.c_[Z[C[i]],train_without_label[i]]
            Output[C[i]]=np.c_[Output[C[i]],train_with_label[i]]     
        for k in range(K):
            Z[k]=Z[k].T
        for k in range(K):
            Centroids[:,k]=np.mean(Z[k],axis=0)
        return Output

def print_number_in_all_cluster(Output):
    for i  in range(0,10):
        print("cluster "+str(i))
        print("Number of elements in a cluster: " + str(Output[i].shape[1]))
        print(Output[i][no_of_features,:])

def visualize_50_digits_in_cluster(Output):
    for j in range(0,10):
        plt.figure(figsize=(30,10))
        print('Cluster ' + str(j))
        numOfRows = 50
        print(str(Output[j].shape[1]) + " elements")
        for i in range(0,50):
            plt.subplot(5+1,10,i+1)
            image = Output[j][0:no_of_features,i]
            image = image.reshape(8,8)
            plt.imshow(image,cmap='gray')
            plt.axis('off')
        plt.show()

def measuring_accuracy(Centroids,test_data,test_label):
    #using the centroids calculated from training data we calculate the least distance using eucladian norm 
    # the point whose distance is least to a centroid the point is assigned to that centroid
    number_of_test_data = test_data.shape[0]
    EuclidianDistance=np.array([]).reshape(number_of_test_data,0)
    for k in range(K):
        tempDist=np.sum((test_data-Centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
    predicted_value=np.argmin(EuclidianDistance,axis=1) #tells to which cluster data point is assigned
    number_of_correct_prediction = 0
    for i in range(0,predicted_value.shape[0]):
        if(predicted_value[i] == test_label[i]): #comapring the cluster value to actual label we find number of correct prediction
            number_of_correct_prediction = number_of_correct_prediction + 1
    accuracy = number_of_correct_prediction/number_of_test_data * 100
    print('Accurcy in test Dataset:')
    print(accuracy)

train_with_label,train_without_label,test_data,test_label = test_train_splitter(pca_reduced_dataset)
m = train_without_label.shape[0] #no. of points in training dataset
no_of_features = train_without_label.shape[1] #number of features
K=10 #number of clusters
Centroids = centroid_initialization(train_with_label,train_without_label)
Output = kmeans_algorithm(train_with_label,train_without_label,Centroids)

print_number_in_all_cluster(Output)

measuring_accuracy(Centroids,test_data,test_label)

