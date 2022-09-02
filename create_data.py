import matplotlib.pyplot as plt
import numpy as np

def create_data(p,n,mA,sigmaA,mB,sigmaB,train_perc, plot = True):
    # Create classA and classB data
    classA=np.zeros((p,n))
    classB=np.zeros((p,n))
    classA[0]= np.concatenate((np.random.randn(1,int(0.5*n))*sigmaA-mA[0],np.random.randn(1,int(0.5*n))*sigmaA+mA[0]),axis=1)
    classB[0]= np.random.randn(1,n)*sigmaB +mB[0]
    classA[1]= np.random.randn(1,n)*sigmaA +mA[1]
    classB[1]= np.random.randn(1,n)*sigmaB +mB[1]

    # Transpose to shuffle data based on rows
    classA=classA.T
    classB=classB.T
    np.random.shuffle(classA)
    np.random.shuffle(classB)
    classA=classA.T
    classB=classB.T

    # Select train and test data
    classA_train=classA[:,:int(train_perc*n)]
    classB_train=classB[:,:int(train_perc*n)]
    classA_test=classA[:,int((1-train_perc)*n):]
    classB_test=classB[:,:int((1-train_perc)*n):]

    # Create pattern and target train
    patterns_train=np.concatenate((classA_train,classB_train),axis=1)
    targets_train=np.concatenate((np.ones(int(np.shape(classA_train)[1])),-np.ones(int(np.shape(classB_train)[1]))))
    s = np.arange(patterns_train.shape[1])
    np.random.shuffle(s)
    patterns_train=patterns_train[:,s]
    targets_train=targets_train[s]

    # Create pattern and target test
    patterns_test=np.concatenate((classA_test,classB_test),axis=1)
    patterns_test=np.concatenate((patterns_test,np.ones((1,np.shape(patterns_test)[1]))))
    targets_test=np.concatenate((np.ones(int(np.shape(classA_test)[1])),-np.ones(int(np.shape(classB_test)[1]))))
    s = np.arange(patterns_test.shape[1])
    np.random.shuffle(s)
    patterns_test=patterns_test[:,s]
    targets_test=targets_test[s]

    if plot:
        plt.plot(classA_train[0],classA_train[1], '*', color = 'red')
        plt.plot(classB_train[0],classB_train[1], '*', color = 'green')
        plt.plot(classA_test[0], classA_test[1], '*', color='yellow')
        plt.plot(classB_test[0], classB_test[1], '*', color='blue')
        plt.show()

    return patterns_train, targets_train, patterns_test, targets_test


