"""
Created on Sun May 31 11:12:25 2020

@author: Jama Hussein Mohamud
"""
import numpy as np
import pandas as pd
import cvxopt
import scipy.sparse as sp

# Kernels

def one_hot(seq):
    dic = {'A':[1, 0, 0, 0], 'C':[0, 1, 0, 0], 'G':[0, 0, 1, 0], 'T':[0, 0, 0, 1]}
    representation_vector = []
    for char in seq:
        representation_vector += dic[char]
    return representation_vector

def apply(data, column='seq', func=one_hot):
    print('----- appying function to the dataframe ------')
    data = data[column].apply(lambda x: func(x))
    data = pd.DataFrame(data.to_list())
    return data

def rbf_kernel(X1, X2, sigma=10):
    if isinstance(X1, pd.DataFrame) and isinstance(X2, pd.DataFrame):
        X1, X2 = np.array(X1), np.array(X2)
    X2_norm = np.sum(X2 ** 2, axis = -1)
    X1_norm = np.sum(X1 ** 2, axis = -1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K

# get k subsequences for our spectrum kernel and bag of words based kernel
def getKmers(sequence, k=3):
    return [sequence[x:x+k].lower() for x in range(len(sequence) - k + 1)]

# get possible subsequences of ['a', 'g', 'c', 't'] 
def poss_seq(k):
    poss_kmers = []
    set = ['a', 'g', 'c', 't']
    n = len(set)  
    def allKLengthRec(poss_kmers, set, prefix, n, k): 
        if (k == 0) : 
            poss_kmers.append(prefix)
            return poss_kmers
        for i in range(n): 
            newPrefix = prefix + set[i] 
            allKLengthRec(poss_kmers, set, newPrefix, n, k - 1)
            
    allKLengthRec(poss_kmers, set, "", n, k) 
    return poss_kmers

# base 4 subseqences for spectrum kernel
def count_base(sequence, k):
    keys = poss_seq(k)
    getKmer = getKmers(sequence, k)
    dic = {i : 0 for i in keys}
    dic1 = {'a': 0, 'g': 1, 'c': 2, 't':3}
    for key in getKmer:
        char = list(key)
        s = 0
        for i,elt in enumerate(char):
            s += dic1[elt]*4**(k-1-i)
        dic[key] = s
    return list(dic.values())

# occurance of each sub sequence BOW
def count_occur(sequence, k):
    keys = poss_seq(k)
    getKmer = getKmers(sequence, k)
    dic = {i : 0 for i in keys}
    for key in getKmer:
        dic[key] +=1
    return list(dic.values())

def apply_store_occur(X,k):
    X = np.array([list(i) for i in X['seq'].apply(lambda x: count_occur(x, k))])
    X = sp.csr_matrix(X)
    return X
def apply_store_baseK(X,k):
    X = np.array([list(i) for i in X['seq'].apply(lambda x: count_base(x, k))])
    X = sp.csr_matrix(X)
    return X

def word_to_index_missmatch(word, m=1):
    BASE = {'A':0, 'C':1, 'G':2, 'T':3}
    base_codes = []
    for i, carac in enumerate(word):
        base_codes += [BASE[carac] * (4 ** i)]
    
    base_index = sum(base_codes)
    indices = [base_index]
    if m >= 1:
        for i, code  in enumerate(base_codes):
            for j in range(4):
                if j != BASE[word[i]]:
                    index = base_index - code + j * 4**i
                    indices.append(index)
                    if m >=2:
                        for i_, code_  in enumerate(base_codes):
                            for j_ in range(4):
                                if i_!=i and j_ != BASE[word[i_]] and j != BASE[word[i]]:
                                    index = base_index - code + j * 4**i - code_ + j_ * 4**i_
                                    indices.append(index)                   
    return indices

def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M, tc='d') if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    #cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
    return np.array(solution['x']).flatten()

solve_qp = cvxopt_qp

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# occurance based gram matrix
def Gram_matrix(X1, X2, k):
    X1 = apply_store_occur(X1,k)
    X2 = apply_store_occur(X2,k)
    return X1 @ X2.T
# base 4 expansion based gram matrix
def Gram_matrix_base(X1, X2, k):
    X1 = apply_store_baseK(X1,k)
    X2 = apply_store_baseK(X2,k)
    return X1 @ X2.T

#  normalizing matrix
def normalize(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    return (matrix - mean)/std

class Multi_SpectrumSVM(object):
    def __init__(self, C=1, k=None, coeff = None, matrix_kernel=Gram_matrix, normalise=True):
        self.C = C
        self.matrix_kernel = Gram_matrix
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        
        self.k = k
        self.coeff = coeff
        self.normalise = normalise
    def fit(self, X, y):
        n_samples = len(X)
        self.train = X
        if isinstance(self.k, list):
            K1 = np.zeros((n_samples,n_samples))
            for i, el in enumerate(self.k):
                kernel = self.matrix_kernel(X,X, el).todense()
                K1 += self.coeff[i]*kernel
        else:
            K1 = self.matrix_kernel(X,X, self.k).todense()
        if self.normalise:
            K1 = normalize(np.array(K1))
        else:
            K1 = np.array(K1)
        
        P = np.outer(y,y) * K1
        q = np.ones(n_samples) * -1
        A = y.reshape(1,n_samples)
        b = np.array([[0.0]])

        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = np.vstack((tmp1, tmp2))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = np.hstack((tmp1, tmp2))

        # solve QP problem
#        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
                    
        # Lagrange multipliers
        a = cvxopt_qp(P, q, G, h, A, b) #np.ravel(solution['x'])
        
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K1[ind[n],sv])
        self.b /= len(self.a)

    def project(self, X):
        y_predict = np.zeros(len(X))
        if isinstance(self.k, list):
            big_K = np.zeros((len(self.train),len(X)))
            for i, el in enumerate(self.k):
                kernel = self.matrix_kernel(self.train, X, el).todense()
                big_K += self.coeff[i]*kernel
        else:
            big_K = self.matrix_kernel(self.train, X, self.k).todense()
        
        if self.normalise:
            big_K = normalize(np.array(big_K))
        else:
            big_K = np.array(big_K)
        
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv.Id):
                s += a * sv_y * big_K[sv, i]
            y_predict[i] = s
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
    
class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    
    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'rbf': rbf_kernel,
    }
    def __init__(self, kernel='rbf', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf':
            params['sigma'] = kwargs.get('sigma', 1.)
        # if self.kernel_name == 'custom_kernel':
        #     params['parameter_1'] = kwargs.get('parameter_1', None)
        #     params['parameter_2'] = kwargs.get('parameter_2', None)
        return params

    def fit(self, X, y, **kwargs):
        return self
        
    def decision_function(self, X):
        pass

    def predict(self, X):
        pass

class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y
        n = len(self.y_train)
        
        A = self.kernel_function_(X, X, **self.kernel_parameters)
        A[np.diag_indices_from(A)] += self.lambd * n
        # self.alpha = (K + n lambda I)^-1 y
        self.alpha = np.linalg.solve(A , self.y_train)

        return self
    
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return K_x.dot(self.alpha )
    def predict(self, X):
        return self.decision_function(X)
    
class WeightedKernelRidgeRegression(KernelRidgeRegression):
    '''
    Weighted Kernel Ridge Regression
    
    This is just used for the KernelLogistic following up
    '''
    def fit(self, K, y, sample_weights=None):

        self.y_train = y
        n = len(self.y_train)
        
        w = np.ones_like(self.y_train) if sample_weights is None else sample_weights
        W = np.diag(np.sqrt(w))
        
        A = W.dot(K).dot(W)
        A[np.diag_indices_from(A)] += self.lambd * n
        # self.alpha = W (K + n lambda I)^-1 W y
        self.alpha = W.dot(np.linalg.solve(A , W.dot(self.y_train)))

        return self

class KernelLogisticRegression(KernelMethodBase):
    '''
    Kernel Logistic Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelLogisticRegression, self).__init__(**kwargs)

    def fit(self, X, y, max_iter=100, tol=1e-5):
        
        X = apply(X, column='seq', func=one_hot)
        self.X_train = X
        self.y_train = y
        
        K = self.kernel_function_(X, X, **self.kernel_parameters)
        
        # IRLS
        WKRR = WeightedKernelRidgeRegression(
            lambd=self.lambd,
            kernel=self.kernel_name,
            **self.kernel_parameters
        )
        # Initialize
        alpha = np.zeros_like(self.y_train)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            f = K.dot(alpha_old)
            w = sigmoid(f) * sigmoid(-f)
            z = f + y / sigmoid(-y*f)
            alpha = WKRR.fit(K, z, sample_weights=w).alpha
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol:
                break
        self.n_iter = n_iter
        self.alpha = alpha

        return self


    def decision_function(self, X):
        X = apply(X, column='seq', func=one_hot)
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        #probability of y = 1 between 0 and 1
        return sigmoid(K_x.dot(self.alpha))

    def predict(self, X):
        # predicted_classes = np.round(self.decision_function(X))
        proba = self.decision_function(X)
        predicted_classes = np.where(proba < 0.5, -1, 1)
        return predicted_classes

def svm_dual_soft_to_qp_kernel(K, y, C=1):
    n = K.shape[0]
    assert (len(y) == n)
        
    # Dual formulation, soft margin
    P = np.diag(y).dot(K).dot(np.diag(y))
    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    P += eps * np.eye(n)
    q = - np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    A = y[np.newaxis, :]
    A = A.astype('float')
    b = np.array([0.])
    return P, q, G, h, A, b

# count subsequences for mismatch kernel
def count_occurences(seq, k=3, m=0):
    index_size = 4 ** k
    counts = np.zeros(index_size, dtype=int)
        
    for i in range(len(seq) - k + 1):
        word = seq[i:(i+k)]
        index = word_to_index_missmatch(word, m)
        counts[index] += 1
    return sp.csr_matrix(counts)

def mismatch_gram_matrix(X1,X2, k, m):
    print(' WAIT------ getting mismatch kernel representation of the data ------ be patient')
    X1 = np.array(X1.seq.parallel_apply(count_occurences, k=k, m=m))
    X2 = np.array(X2.seq.parallel_apply(count_occurences, k=k, m=m))
    X1 = sp.vstack(X1)
    X2 = sp.vstack(X2)
    K  = (X1@X2.T).toarray()
    K = (K - K.mean()) / K.std() 
    return K

class KernelSVM(object):
    '''
    Kernel SVM Classification
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, C=0.1, k = 3, m = 1):
        self.C = C
        self.kernel_mismatch = mismatch_gram_matrix      
        self.k = k
        self.m = m

    def fit(self, X, y, tol=1e-5):
        #n, p = X.shape
        #assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        # Kernel matrix
        K = self.kernel_mismatch(self.X_train, self.X_train, self.k, self.m)
        
        # Solve dual problem
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.C))
        
        # Compute support vectors and bias b
        sv = np.logical_and((self.alpha > tol), (self.C - self.alpha > tol))
        self.bias = y[sv] - K[sv].dot(self.alpha * y)
        #print(self.bias)
        self.bias = self.bias.mean()

        self.support_vector_indices = np.nonzero(sv)[0]

        return self
        
    def decision_function(self, X):
        K_x = self.kernel_mismatch(X, self.X_train, self.k, self.m)
        return K_x.dot(self.alpha * self.y_train) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    