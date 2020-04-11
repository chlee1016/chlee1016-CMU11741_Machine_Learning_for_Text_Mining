import sys

from sklearn.datasets import load_svmlight_file
from svm import *
from scipy import sparse
import matplotlib.pyplot as plt

def main():
    # read the train file from first arugment
    # print("run main.py")
    train_file = sys.argv[1]

    # read the test file from second argument
    test_file = sys.argv[2]
    car = train_file[8:15]

    # You can use load_svmlight_file to load data from train_file and test_file
    X_train, y_train = load_svmlight_file(train_file)
    X_test, y_test = load_svmlight_file(test_file)

    Bias_train = np.ones((np.size(X_train, axis=0),1))
    Bias_train = sparse.csr_matrix(Bias_train)
    Bias_test = np.ones((np.size(X_test, axis=0),1))
    Bias_test = sparse.csr_matrix(Bias_test)

    X_train = sparse.hstack([X_train, Bias_train])
    X_test = sparse.hstack([X_test, Bias_test])

    n = np.size(X_train, axis=1)

    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)

    svm = Svm(n, cartegory=car)
    hist_sgd = svm.train_sgd(X_train, y_train, X_test, y_test)
    hist_ner = svm.train_ner(X_train, y_train, X_test, y_test)
    print(hist_sgd.history)
    print(hist_ner.history)
    # You can use cg.ConjugateGradient(X, I, grad, lambda_)

    plt.figure(1)

    plt.subplot(131)
    plt.plot(hist_sgd.history['sgd_time'], hist_sgd.history['sgd_rfvd'], linestyle='--')
    plt.plot(hist_ner.history['ner_time'], hist_ner.history['ner_rfvd'], linestyle=':')
    plt.title('Relative function value difference (RFVD)')
    plt.ylabel('RFVD')
    plt.xlabel('Training time')
    plt.legend(['SGD', 'Newton-Raphson'], loc='upper right')

    plt.subplot(132)
    plt.plot(hist_sgd.history['sgd_time'], hist_sgd.history['sgd_gnorm'], linestyle='--')
    plt.plot(hist_ner.history['ner_time'], hist_ner.history['ner_gnorm'], linestyle=':')
    plt.title('Gradient norm')
    plt.ylabel('||f(w)||_2')
    plt.xlabel('Training time')
    plt.legend(['SGD', 'Newton-Raphson'], loc='upper right')

    plt.subplot(133)
    plt.plot(hist_sgd.history['sgd_time'], hist_sgd.history['sgd_acc'], linestyle='--')
    plt.plot(hist_ner.history['ner_time'], hist_ner.history['ner_acc'], linestyle=':')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training time')
    plt.legend(['SGD', 'Newton-Raphson'], loc='lower right')

    plt.show()


# Main entry point to the program
if __name__ == '__main__':
    main()
