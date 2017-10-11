import numpy as np
def extractPCA(data, In = 0.85):
    inv = False
    if (data.shape[0] > data.shape[1]):
        inv = True
        data = data.T
    c = np.cov(data)
    eigvals, eigvect = np.linalg.eig(c)
    eigvals = eigvals.real
    ind = np.argsort(eigvals)
    ind = ind[::-1]
    eigvals = eigvals[ind]
    eigvect = eigvect[:,ind]
    cumEigvals = np.cumsum(eigvals) / np.sum(eigvals)
    idx = np.where(cumEigvals > In)[0]
    V = []
    if (In < 1):
      V = eigvect[:,:idx[0]]
    else:
      V = eigvect[:,:In]
    pca_data = V.T.dot(data)
    if(inv):
        return pca_data.T
    return pca_data
def extractPCAEIGV(data, In = 0.85):
    inv = False
    if (data.shape[0] > data.shape[1]):
        inv = True
        data = data.T
    c = np.cov(data)
    #print(c.shape)
    eigvals, eigvect = np.linalg.eig(c)
    eigvals = eigvals.real
    ind = np.argsort(eigvals)
    ind = ind[::-1]
    eigvals = eigvals[ind]
    eigvect = eigvect[:,ind]
    cumEigvals = np.cumsum(eigvals) / np.sum(eigvals)
    idx = np.where(cumEigvals > In)[0]
    V = eigvect[:,:idx[0]]
    #if (inv): 
    return V
    #return V.T
def extractLDA(data, labels, nvect = 15):
    inv = False
    if (data.shape[0] > data.shape[1]):
        inv = True
        data = data.T
    classes = np.unique(labels)
    count = np.bincount(labels) 
    means = np.array([(data[:,np.where(labels == classes[i])[0]].mean(1)) for i in range(classes.shape[0])]).T
    scatters = np.array([np.cov(data[:,np.where(labels == classes[i])[0]]) for i in range(classes.shape[0])])
    scatterw = np.sum(scatters[i] for i in range(classes.shape[0]))
    scatterb =  np.cov(means)
    scatterwinvB = np.dot(np.linalg.pinv(scatterw).real, (scatterb))
    eigvals, eigvect = np.linalg.eig(scatterwinvB)
    idx = eigvals.real.argsort()[::-1]
    eigvals= eigvals[idx]
    eigvect = eigvect[:, idx]
    V =  eigvect[:, :nvect]
    return V
    #lda_data = V.T.dot(data)
    #if(inv):
     #   return lda_data.T
    #return lda_data
