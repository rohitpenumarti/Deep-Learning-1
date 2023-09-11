import numpy as np
import matplotlib.pyplot as plt
import json
import h5py

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x)))

def predict(W_, b_, X, h, out):
    data = []
    for i, x in enumerate(X):
        res = {}
        res["index"] = i
        
        a = []
        for j, (W, b) in enumerate(zip(W_, b_)):
            if j == 0:
                a.append(h(np.matmul(W, x)+b))
            elif j == len(W_)-1:
                a.append(out(np.matmul(W, a[j-1])+b))
            else:
                a.append(h(np.matmul(W, a[j-1])+b))
        
        res["activations"] = a[-1].astype(str).tolist()
        res["classification"] = int(np.argmax(a[-1]))
        data.append(res)
    
    with open("result.json", "w") as f:
        f.write(json.dumps(data))

    return data

def accuracy(res, y_true):
    y_without_one_hot = np.argmax(y_true, axis=1)
    y_pred = np.zeros(y_without_one_hot.shape)
    for i, r in enumerate(res):
        y_pred[i] = r["classification"]

    correct = 0
    for yp, yt in zip(y_pred, y_without_one_hot):
        if yp == yt:
            correct += 1

    return correct/np.size(y_without_one_hot)

def main():
    params_file_path = "mnist_network_params.hdf5"
    input_size = 784
    hidden_neurons1 = 200
    hidden_neurons2 = 100
    output_size = 10

    with h5py.File(params_file_path, 'r') as hf:
        b1 = hf["b1"][:]
        W1 = hf["W1"][:]
        b2 = hf["b2"][:]
        W2 = hf["W2"][:]
        b3 = hf["b3"][:]
        W3 = hf["W3"][:]

    assert b1.shape[0] == hidden_neurons1, 'Error: wrong dimensions'
    assert W1.shape == (hidden_neurons1, input_size), 'Error: wrong dimensions'
    assert b2.shape[0] == hidden_neurons2, 'Error: wrong dimensions'
    assert W2.shape == (hidden_neurons2, hidden_neurons1), 'Error: wrong dimensions'
    assert b3.shape[0] == output_size, 'Error: wrong dimensions'
    assert W3.shape == (output_size, hidden_neurons2), 'Error: wrong dimensions'

    test_fpath = "mnist_testdata.hdf5"

    with h5py.File(test_fpath, 'r') as hf:
        X_test = hf["xdata"][:]
        y_test = hf["ydata"][:]

    W = [W1, W2, W3]
    b = [b1, b2, b3]

    pred_data = predict(W, b, X_test, ReLU, softmax)
    test_accuracy = accuracy(pred_data, y_test)
    print(f'The total number of correctly classified points is: {int(test_accuracy*y_test.shape[0])}, for a test accuracy of: {test_accuracy}')

    num_correct = 0
    num_incorrect = 0
    while num_correct+num_incorrect < 6:
        rand_idx = np.random.randint(0, np.size(pred_data))
        y_without_one_hot = np.argmax(y_test, axis=1)

        if pred_data[rand_idx]["classification"] == y_without_one_hot[rand_idx]:
            num_correct += 1
        else:
            num_incorrect += 1

        plt.figure()
        plt.imshow(X_test[rand_idx].reshape(28,28))
        plt.savefig(f"Image{rand_idx}.png")

if __name__ == "__main__":
    main()