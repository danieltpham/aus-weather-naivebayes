import numpy as np
import pandas as pd
import json

def gaussian_pdf(x, mean, std):
    '''
    Return the gaussian pdf for a specific value x, given mean & std
    '''
    if (std==0): 
        # All instances are identical => not useful for classification
        return 1
    exponent = np.exp(-1/2*((x - mean)/std)**2)
    return exponent / (np.sqrt(2 * np.pi) * std)

class Matrix:
    '''
    This class stores the basic information about the dataset (X, y)
    The .prior() method returns the prior probabilities of the target vector
    '''
    def __init__(self, X, y):
        self.att_mat = X.copy() # Attribute matrix
        self.target = y # Target column
        self.labels = self.target.unique() # Classes
        self.n_inst = X.shape[0] # Number of instances / rows
        self.n_attr = X.shape[1] # Number of attributes / cols
    
    def prior(self):
        # This method returns the Priors of the classes
        # stored as a dict of {class: prior}
        prior = dict()
        for label in self.labels:
            prob = len(self.target[self.target==label])/self.n_inst
            prior[label] = prob
        return prior
        
class NominalVector:
    '''
    This class stores the nominal vector and calculates its cond. likelihood
    Input:  X: attribute matrix
            index: numeric index of the nominal vector
            k: additive constant for smoothing 
               (default = 1, inherit from NaiveBayesModel class)
    '''
    def __init__(self, matrix, index, k):
        self.x = matrix.att_mat.iloc[:,index] # Nominal attribute column
        self.llh = self.likelihood(matrix, k) # Likelihood 2D dict
    
    def likelihood(self, matrix, k):
        # This method returns a dict of dict for frequency with smoothing
        # with level of indexing: count = llh[class label][attribute value]
        # and then divide by the total count of instances adjusted by smoothing
        # to get likelihood with the same level of indexing
        
        # Initialise first level of dict
        llh = dict.fromkeys(matrix.labels)
        
        # Frequency count for each class label is initialised as `k`
        for label in matrix.labels:
            llh[label] = {i:k for i in self.x.unique()}
            
        # Loop through attribute column and count frequencies
        for i in range(matrix.n_inst):
            label = matrix.target[i]
            attr_val = self.x.iloc[i:i+1,].values[0]
            if not isinstance(attr_val, float): # Ignoring NaN which are of type `float`
                llh[label][attr_val] += 1
        
        # Calculating probabilities from frequencies
        for label in matrix.labels:
            for attr in self.x.unique():
                llh[label][attr] /= (len(matrix.target[matrix.target==label])+len(self.x.unique()))
        return llh

class NumericVector:
    '''
    This class stores the numeric vector and calculates its Gaussian likelihood
    Input:  X: attribute matrix
            index: numeric index of the numeric vector
    '''
    def __init__(self, matrix, index):
        self.x = matrix.att_mat.iloc[:,index] # Numeric attribute column
        self.llh = self.likelihood(matrix) # Likelihood 2D dict
        
    def likelihood(self, matrix):
        # This method returns a dict of dict for Gaussian likelihood
        # with level of indexing: {mu, sigma} = llh[class label]
        
        # Initialise first level of dict
        llh = dict.fromkeys(matrix.labels)
        
        # Calculate mean & std for each class label, ignoring missing instances
        for label in matrix.labels:
            # Select instances with that label
            idx = matrix.target.index[matrix.target == label].tolist()
            llh[label] = {"mu": np.nanmean(self.x.iloc[idx]),
                               "sigma": np.nanstd(self.x.iloc[idx],ddof=1)}
        return llh

class NaiveBayesModel:
    '''
    Input:  X: attribute matrix
            y: target vector
            attr_types: updated accompanied attribute code vector
            nominal_smoothing: additive constant for smoothing nominal frequency
                               default to `1` as Laplace Smoothing
    Output Class Attributes:
           .prior : Prior vector stored as dict
           .labels: Class label stored as list
           .attr_types: Copy of `attr_types` from input
           .params: list of dict of dict of likelihoods for each attribute
    '''
    def __init__(self, X, y, attr_types, nominal_smoothing=1):
        m = Matrix(X, y)
        self.prior = m.prior()
        self.labels = m.labels
        self.attr_types = attr_types
        
        # Loop through the list of attributes 
        # and store the likelihood vector into parameter list
        self.params = []
        for i in range(m.n_attr):
            if (attr_types[i] == 0  or attr_types[i] == 1): # Nominal & Ordinal Case
                self.params.append(NominalVector(m, i, nominal_smoothing).llh)
            elif (attr_types[i] == 2): # Numeric Case
                self.params.append(NumericVector(m, i).llh)

    def load_model(self, model_name):
        # Load existing model with params
        with open(model_name+'.json') as f:
            saved_model = json.load(f)
        self.prior = saved_model['prior']
        self.labels = list(saved_model['labels'])
        self.attr_types = saved_model['attr_types']
        self.params = saved_model['params']
        
    def save_model(self, model_name):
        # Save trained model to json
        export = {'prior': self.prior,
            'labels': self.labels.tolist(),
            'attr_types': self.attr_types,
            'params': self.params}
        with open(model_name+'.json', 'w') as f:
            json.dump(export, f)