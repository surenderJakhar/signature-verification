import tensorflow as tf
import pandas as pd
import numpy as np
from time import time
import test
import dataset

n_input = 9
#dataset.makeCSV()

train_person_id = input("Enter person's id : ")
img_path = input("Enter path of signature image : ")

f_count=0

g_count=0
for i in range(0,5):
    
    test_image_path='data/'+img_path+'/'+train_person_id+train_person_id+'_00'+str(i)+'.png'
    train_path = '~/Signature-Verification/data/Features/Training/training_'+train_person_id+'.csv'
    test.testing(test_image_path)
    test_path = '~/Signature-Verification/data/TestFeatures/testcsv.csv'

    def readCSV(train_path, test_path, type2=False):
        # Reading train data
        df = pd.read_csv(train_path, usecols=range(n_input))
        train_input = np.array(df.values)
        train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
        df = pd.read_csv(train_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_train = np.eye(2)[correct]      # Converting to one hot
        # Reading test data
        df = pd.read_csv(test_path, usecols=range(n_input))
        test_input = np.array(df.values)
        test_input = test_input.astype(np.float32, copy=False)
        if not(type2):
            df = pd.read_csv(test_path, usecols=(n_input,))
            temp = [elem[0] for elem in df.values]
            correct = np.array(temp)
            corr_test = np.eye(2)[correct]      # Converting to one hot
        if not(type2):
            return train_input, corr_train, test_input, corr_test
        else:
            return train_input, corr_train, test_input
    
    tf.compat.v1.reset_default_graph()
    # Parameters
    learning_rate = 0.001
    training_epochs = 1000
    display_step = 1
    
    # Network Parameters
    n_hidden_1 = 7 # 1st layer number of neurons    
    n_hidden_2 = 10 # 2nd layer number of neurons
    n_hidden_3 = 30 # 3rd layer
    n_classes = 2 # no. of classes (genuine or forged)

    # tf Graph input
    X = tf.compat.v1.placeholder("float", [None, n_input])
    Y = tf.compat.v1.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], seed=1)),
        'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random.normal([n_hidden_1, n_classes], seed=2))
    }
    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden_1], seed=3)),
        'b2': tf.Variable(tf.random.normal([n_hidden_2])),
        'b3': tf.Variable(tf.random.normal([n_hidden_3])),
        'out': tf.Variable(tf.random.normal([n_classes], seed=4))
    }

    
    # Create model
    def multilayer_perceptron(x):
        layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer

    loss_op = tf.reduce_mean(tf.math.squared_difference(logits, Y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # For accuracies
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Initializing the variables
    init = tf.compat.v1.global_variables_initializer()

    def evaluate(train_path, test_path, type2=False): 
        global f_count
        global g_count  
        if not(type2):
            train_input, corr_train, test_input, corr_test = readCSV(train_path, test_path)
        else:
            train_input, corr_train, test_input = readCSV(train_path, test_path, type2)
        ans = 'Random'
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                # Run optimization op (backprop) and cost op (to get loss value)
                _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
                if cost<0.0001:
                    break
#                 # Display logs per epoch step
                if epoch % 999 == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost))
            print("Optimization Finished!")
            
            # Finding accuracies
            accuracy1 =  accuracy.eval({X: train_input, Y: corr_train})
            print("Accuracy for train:", accuracy1)
#             print("Accuracy for test:", accuracy2)
            if type2 is False:
               # accuracy2 =  accuracy.eval({X: test_input, Y: corr_test})
                return accuracy1#, accuracy2
            else:
                prediction = pred.eval({X: test_input})
                if prediction[0][1]>prediction[0][0]:
                    print('Genuine Signature')
                    g_count=g_count+1
                    return True
                else:
                    print('Forged Signature')
                    f_count=f_count+1
                    return False

    evaluate(train_path, test_path, type2=True)
print("Accuracy : ",g_count/5)
