#First import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

my_data = pd.read_csv('data.csv')
my_data.head()
my_data.info()

#The line below should display (569, 33)

my_data.shape
#To remove the null values from the dataset, we use dropna(), and axis=1 means to remove them from columns.

my_data.dropna(axis=1, inplace=True)
#The id column is also unnecessary so we removed it from the dataset using my_data.drop() function.

my_dt= my_data.drop(['id'], axis = 1)

my_dt.head(3)
#Every field in the dataset contributes equally to predicting the outcome diagnosis. The cancer diagnosis is mainly two, whether it’s benign or malignant. The below code displays the two diagnoses.
The_M = my_dt[my_data.diagnosis == "M"]

The_M.head()

The_B = my_dt[my_data.diagnosis == "B"]

The_B.head(6)
#The scatter plot is used here to display the texture mean and radius mean of Benign and Malignant.

#Plot both diagnosis Benign and Malignant 

plt.title("Benign Tumor VS Malignant")

plt.xlabel("Radius_Mean")

plt.ylabel("Texture_Mean")

plt.scatter(The_M.radius_mean, The_M.texture_mean, color = "blue", label = "Malignant", alpha = 0.4)

plt.scatter(The_B.radius_mean, The_B.texture_mean, color = "orange", label = "Benign", alpha = 0.4)

plt.legend()

plt.savefig("importance graph 4", facecolor='w', bbox_inches='tight',

            pad_inches=0.3, transparent=True)

plt.show()
#Now let’s implement the decision tree classifier using the Scikit-learn library.

my_data.diagnosis = [1 if i == "M" else 0 for i in my_data.diagnosis]

x_val = my_data.drop(["diagnosis"], axis = 1)
y_val = my_data.diagnosis.values
#The min-max normalization is used to smooth the data and transform large values into small scales to process the data easily and get high accuracy.
#Min_Max Normalization:
my_x= (x_val - np.min(x_val)) / (np.max(x_val) - np.min(x_val))
my_x.head()
#Now split the dataset into training and testing sets. We have used 40% for testing and 60% for training.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(my_x,y_val,test_size=0.4,random_state = 41)
#import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
#Setup arrays to store training and test accuracies

my_neighbors = np.arange(1,10)

train_acc =np.empty(len(my_neighbors))

test_acc = np.empty(len(my_neighbors))

for i,k in enumerate(my_neighbors):

    #Setup a knn classifier with k neighbors

    knn_model = KNeighborsClassifier(n_neighbors=k)  

    #Fit the model

    knn_model.fit(X_train, y_train)

    #Compute accuracy on the training set

    train_acc[i] = knn_model.score(X_train, y_train)

    #Compute accuracy on the test set

    test_acc[i] = knn_model.score(X_test, y_test)
