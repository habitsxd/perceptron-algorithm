
import numpy as np

def main():
    data_train = np.genfromtxt('data-train.csv',delimiter = ',')#generate array from training data set
    
    data_test = np.genfromtxt('data-test.csv',delimiter = ',') #generate array from test data
   
    weights = np.zeros(len(data_train[0])) #initialize weight vector at 0
    
    bias = 0 #initialize bias, arbitrary number
    
    train(data_train,weights) #train the perceptron
    
    print('The test data run results: ')
    print()
    perceptron(data_test,weights) #predict new values based on the test data
    
    
def predict(activation):
    #compare weighted sum and bias and return a binary classification
    if activation >= 0:
        return 1
    else:
        return -1

def train(data_set,weights):
    hits = 0 #counts number of times we get a successful classification
    count = 0 #used to count how many times we pass through the data set
    while hits != len(data_set):#while number of successful classifications does not equal the length of the data set
        hits = 0 #reset hits when we get back to the beginning of the data set
        x_vector = [1]
        for i in range(len(data_set)): #for each array in data set
            weighted_sum = 0 #reset the sum as we move to the next element in the dataset 
            for a in range(len(data_set[i] - 1)): #create new x_vector containing a 1 at the zeroth element
                x_vector.append(data_set[a + 1])
            for j in range(len(weights)): #multiply the weights by the dataset vector
                weighted_sum += (weights[j] * x_vector[j])
            activation = weighted_sum + weights[0] #compute activation equation to test classification
            if predict(activation) != data_set[i][0]: #if misclassified
                for k in range(len(weights)): #update
                    weights[k] = weights[k] + (data_set[i][0] * x_vector[k])
            else: #if successful classification increase count
                hits += 1
        count += 1
        print("Weights have been updated: " + str(weights)) #these last lines are just for clean output analysis
    print()
    print("Optimal weights found: " + str(weights))
    print("The algorithm passed over the data set " + str(count) + " times.")
    
    
def perceptron(data,weights): 
    #binary classification for new data based on the weights generated from the train method
    
    for i in range(len(data)):
        weighted_sum = 0
        x_vector = [1]
        for a in range(len(data) - 1):
            x_vector.append(data[a + 1])
        for j in range(len(weights)):
            weighted_sum += (weights[j] * x_vector[j])
        activation = weighted_sum + weights[0]
        print(predict(activation))
        


if __name__ == '__main__':
    main()

