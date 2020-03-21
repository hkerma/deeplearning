import matplotlib.pyplot as plt 

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")

def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()

plt.plot(loadList("score_net_weights_layer2.npy"),loadList("fscore_accuracy_layer2.npy"))
plt.plot(loadList("hscore_net_weights_layer2.npy"),loadList("hscore_accuracy_layer2.npy"))
plt.plot(loadList("score_iter_net_weights_layer2.npy"),loadList("score_iter_accuracy_layer2.npy"))
plt.show()