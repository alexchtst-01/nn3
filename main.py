import numpy as np
import matplotlib.pyplot as plt


class modelNeural(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.rand(7, 5) - 0.5
        self.b1 = np.random.rand(7, 1) - 0.5
        self.W2 = np.random.rand(2, 7) - 0.5
        self.b2 = np.random.rand(2, 1) - 0.5
        self.commulative_errors = []
    
    def _param_init(self):
        print(self.W1)
        print(self.b1)
        print(self.W2)
        print(self.b2)
    
    def sgd(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def deriv_sgd(self, Z):
        return (1 - self.sgd(Z)) * (self.sgd(Z))
    
    def softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z))
    
    def deriv_softmax(self, Z):
        return self.softmax(Z) * (1 - self.softmax(Z))
    
    def error_(self, true, pred):
        return(np.sum(np.square(true - pred)))
    
    def _info(self):
        f = open("Readme.txt", 'w')
        f.write("HOW TO USE THIS CLASS\n\n")
        f.write("1. The input X and Y in gradient_descent no need to be transposed\n")
        f.write("2. The input X and Y in predict and predict_unit no need to be transposed\n")
        f.write("3. The input X and Y in Train_model no need to be transposed\n")
        f.write("4. The predict_unit is only for vector param\n")
        f.write("5. The predict_unit is only for Martrix param\n")
        f.write("6. The trainning_itteration is for how many ittereation will happened\n")
        f.write("7. The bacth is for how many error will appears\n")
        f.write("8. If the training\n")
        f.write("9. If trainning_itteration is 1000 and batch is 20 \n\tthen the updates param will updates every 1000/20 = 50 times ones\n\taccording to stochastic gradient descent princip\n")
        f.close()
        print("===THE INSTRUCTION FILE HAS BEEN MADE!!!===")
        print("full instruction in Readme.txt\n")
    
    # forward propegation
    def predict(self, X):
        Z1 = self.W1.dot(X.T) + self.b1
        A1 = self.sgd(Z1)
        
        Z2 = self.W2.dot(A1) + self.b2
        temp_ = Z2.T
        
        Z2 = np.vstack((self.softmax(temp_[0]), self.softmax(temp_[1])))
        for i in range(temp_.shape[1] - 2):
            Z2 = np.vstack((Z2, self.softmax(temp_[i + 2])))
        Z2 = Z2.T
        
        return Z2
    
    def predict_unit(self, x):
        Z1 = self.W1.dot(x.reshape(5, 1)) + self.b1
        a1 = self.sgd(Z1)
        
        Z2 = self.W2.dot(a1) + self.b2
        a2 = self.softmax(Z2)
        
        return a2
    
    # backward propegation
    def gradient_descent(self, x, y):
        y = y.reshape(2, 1)
        
        Z1 = self.W1.dot(x.reshape(5, 1)) + self.b1
        a1 = self.sgd(Z1)
        
        Z2 = self.W2.dot(a1) + self.b2
        a2 = self.softmax(Z2)
        
        C = np.sum(np.square(a2))
        dC_da2 = 2 * (a2 - y)
        
        dW2 = np.ones(shape=(2, 7)) * self.deriv_softmax(Z2) * dC_da2
        db2 = self.deriv_softmax(Z2) * dC_da2
        
        dC_da1 = self.W2.T.dot((dC_da2 * self.deriv_softmax(Z2)))
        
        dW1 = np.ones(shape=(7, 5)) * self.deriv_sgd(Z1) * dC_da1
        db1 = self.deriv_sgd(Z1) * dC_da1
        
        return dW1, db1, dW2, db2
    
    def _update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def Train_model(self, X, Y, batch=20, trainning_itteration=1000):
        size = Y.shape[0]
        error = 0
        group_size = int(trainning_itteration/batch)
        n = 0
        dW1, db1, dW2, db2 = 0, 0, 0, 0
        for itter in range(1, trainning_itteration+1):
            # we are using stochastic gradient descent patterns
            rd_idx = np.random.randint(low=0, high=size)
            rd_x = X[rd_idx]
            rd_y = Y[rd_idx]
            dW1_, db1_, dW2_, db2_ = self.gradient_descent(x=rd_x, y=rd_y)
            dW1 += dW1_
            db1 += db1_
            dW2 += dW2_
            db2 += db2_
                        
            # every batch size we are going to evaluate the error and update the parameters
            if itter % group_size == 0:
                dW1 = dW1/group_size
                db1 = db1/group_size
                dW2 = dW2/group_size
                db2 = db2/group_size
                self._update_params(dW1, db1, dW2, db2)
                for i in range(size):
                    x = X[rd_idx]
                    y = Y[rd_idx]
                    pred = self.predict_unit(x)
                    error += self.error_(true=y, pred=pred)
                n += 1
                print(f"{n}/{batch} the error after {group_size} updates is : [{error}]")
                self.commulative_errors.append(error)
                error = 0
                dW1, db1, dW2, db2 = 0, 0, 0, 0
    
    def save_error_record(self, save=False):
        if save:
            f = open("errorRecord.csv", 'w')
            f.write(f"index,error\n")
            for i, idx in zip(self.commulative_errors, range(len(self.commulative_errors))):
                f.write(f"{idx},{i}\n")
            f.close()
            print("file has been saved as errorRecord.csv")
        else:
            print("file hasm't saved")
    
    def plot_error_(self, save=False):
        plt.plot(self.commulative_errors)
        plt.show()
        if save:
            plt.savefig("./error.png")
            print("file has been saved as error.png")

modelNeural()._info()