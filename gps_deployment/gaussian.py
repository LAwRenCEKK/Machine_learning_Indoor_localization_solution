"""
Author: Jianpeng Liu
Copyrights belong to WiSeR Lab 
Downstream Localiztion model GPs
Reference: https://rse-lab.cs.washington.edu/postscripts/gp-localization-rss-06.pdf
"""

import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import scipy.stats as ss

class GaussianProcess:
    
    def __init__(self):
        self.l = None
        self.sigma_f_squared = None
        self.sigma_n_squared = None
        self.k_x = None #k(x_p,x_q)
        self.cov_y = None
        self.K_inverse = None
        self.training_points = None
        self.training_outputs = None
        self.mean_regressor = None
    

    def set_hyperparameters(self, l, f, n):
        self.l = l
        self.sigma_f_squared = f
        self.sigma_n_squared = n
    

    def get_hyperparameters(self):
        return [self.l, self.sigma_f_squared, self.sigma_n_squared]    
    

    def train_gaussian_model_with_params(self, X, y, params):
        l,sigma_f_2,sigma_n_2 = params
        # print ("Training with Hyperparameters " +  str(params))
        self.training_points = X #location
        self.training_outputs = y #Rssi
                
        #Train model for these parameters
        self.set_hyperparameters(l,sigma_f_2,sigma_n_2)

        #Calculate mean offet        
        self.calculate_mean_offset(X, y)                    
        self.build_covariance_matrix(X)        
        self.calculate_output_covariance()
        self.calculate_covariance_inverse()
    
    
    def train_gaussian_model(self, X, y):
        # X is the location
        # y is the observation
        print ("Training gaussian models to find best parameters ..")        
        best_params = []
        min_error = np.inf
        highest_likelihood = -np.inf
        param_range = self.param_range
        for l in param_range['l']:
            for sigma_f_2 in param_range['sigma_f_squared']:
                for sigma_n_2 in param_range['sigma_n_squared']:       
                    #Split data into training and cross val      
                    k=5
                    kf = KFold(n_splits=k)
                    sum_err = 0
              
                    #Cross Validation      
                    for train_index, val_index in kf.split(X):
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        #Train model for these parameters            
                        self.training_points = X_train
                        self.training_outputs = y_train
                        self.set_hyperparameters(l,sigma_f_2,sigma_n_2)                        

                        #Calculate mean offet    
                        self.calculate_mean_offset(X_train, y_train)
                        self.build_covariance_matrix(X_train)                        
                        self.calculate_output_covariance()
                        self.calculate_covariance_inverse()

                        #Get log likelihood
                        p = self.calculate_log_likelihood() 
                        y_pred = [self.predict_gaussian_value(o) for o in X_val]
                        y_true = y_val
                        e = np.sqrt(mean_squared_error(y_true,y_pred))
                        sum_err +=e
                        
                    avg_err = sum_err/k
                    # print "A", -avg_err , "B:", p
                    #data_fit_error = avg_err #Minimize (+ve)
                    #complexity_term = p     #Maximize (-ve)
                    combined_likelihood = p - avg_err

                    if combined_likelihood > highest_likelihood:
                        best_params = self.get_hyperparameters()
                        min_error = avg_err
                        highest_likelihood = combined_likelihood

        #Train full dataset with best params
        print ("Optimal hyperparameters:" + str(best_params) + "data fit term " + str(min_error) + "Likelihood" + str(highest_likelihood))
        print (str(self.train_gaussian_model_with_params(X,y, best_params)))

    
    def calculate_log_likelihood(self):        
        y = self.training_outputs
        norm_y = y/abs(np.sum(y))                
        
        n = len(y)
        A = np.dot(np.transpose(norm_y),np.dot(self.K_inverse, norm_y))        
        B = 0.5*np.log( np.linalg.det(self.cov_y) )
        C = np.log(np.pi)*n/2
        return -(A+B+C)

    
    #Fit a smooth curve to the wifi means            
    def calculate_mean_offset(self, X, Y):    
        i_centre = np.argmax(Y)
        b = Y[i_centre]
        ap = X[i_centre,:]  
        self.ap_ksum = np.sum(self.build_vector_covariance(ap))

        X_sorted = X[Y.argsort()]
        Y_sorted = sorted(Y)     
        
        m_x = np.mean([(Y_sorted[j]-b)/abs(ap[0]-X_sorted[j,0]) for j in range(10)] )                       
        m_y = np.mean([(Y_sorted[j]-b)/abs(ap[1]-X_sorted[j,1]) for j in range(10)] )        
  
        self.mean_params = [b, ap[0], ap[1], m_x, m_y]        
    
    
    def build_vector_covariance(self, p):
        X = self.training_points
        n = len(X)        
        k_star = np.zeros((n,))
        for i in range(n):
            k_star[i] = self.kernel_exp_squared(p, X[i,:])
        return k_star
    
    
    def build_covariance_matrix(self, points):
        N = len(points)
        self.k_x = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i>j:
                    self.k_x[i,j] = self.k_x[j,i]
                elif i==j:
                    self.k_x[i,i] = self.sigma_f_squared
                else:
                    self.k_x[i,j] = self.kernel_exp_squared(points[i,:], points[j,:])
    
    
    def set_param_ranges(self, a):
        max_l = 12.0
        min_l = 2.0
        l_range = np.arange(min_l+a,max_l+a,1.0)
        f_range = np.arange(1.0+a,5.0+a,1)
        n_range = np.arange(0.5+a/2.0,2.5+a/2.0,0.5)
        self.param_range = {'l': l_range, 'sigma_f_squared': f_range, 'sigma_n_squared': n_range}                
    
    def kernel_exp_squared(self, p, q):
        euclidean_dist = float(np.sum(np.square(p-q)))      
        kernel_output = self.sigma_f_squared * math.exp(-euclidean_dist/ (2*self.l*self.l) )
        return kernel_output
    
            
    def calculate_output_covariance(self):
        if(self.k_x is not None):         
            N = self.k_x.shape[0]    
            noise = np.zeros((self.k_x.shape))
            for i in range(N):
                noise[i,i] = self.sigma_n_squared
            self.cov_y = np.add(self.k_x, noise)
    
    
    def calculate_covariance_inverse(self):
        if self.cov_y is not None:
            self.K_inverse = np.linalg.inv(self.cov_y)
            
    
    def set_prior_covariance(self, y_cov_prior):
        self.prior_covariance = y_cov_prior
    
    
    def get_prior_covariance(self,p):

        return self.sigma_f_squared
        
    
    """
    This method returns the predicted mean and covariance at an arbitrary point p given training data X and ouputs y
    p is given point [x,y]
    X is training data consisting of N inputs
    y is output data consisting of N outputs
    """
    def predict_gaussian_value(self,p):
        #print "Point", p
        k_star = self.build_vector_covariance(p)
        k_star_star = self.get_prior_covariance(p)        
        
        A = np.dot(np.transpose(k_star), self.K_inverse)
        B = np.dot(A,k_star)
        
        y = self.training_outputs
        mean_p = np.dot(A, y)  
        cov_p = np.subtract(k_star_star, B)
        offset = self.predict_mean_offset(p)
        cutoff = self.ap_ksum/2.0
        
        if np.sum(k_star) < cutoff:        
            mean_p += offset*(cutoff-np.sum(k_star))/cutoff   
        result = np.random.normal(mean_p,cov_p)
        
        if(result > -30):
            pass

        return min(-10,max(result , -93.0)) 
    
    
    def predict_gaussian_value2(self,p,ob):
        #print "Point", p
        k_star = self.build_vector_covariance(p)
        k_star_star = self.get_prior_covariance(p)        
        
        A = np.dot(np.transpose(k_star), self.K_inverse)
        B = np.dot(A,k_star)

        y = self.training_outputs
        mean_p = np.dot(A, y)  
        cov_p = np.subtract(k_star_star, B)

        offset = self.predict_mean_offset(p)
        cutoff = self.ap_ksum/2.0
        if np.sum(k_star) < cutoff:        

            mean_p += offset*(cutoff-np.sum(k_star))/cutoff   

        return (ss.norm(mean_p,cov_p).pdf(ob))


    def predict_mean_offset(self, x):

        [b, ap_x, ap_y, m_x, m_y] = self.mean_params

        dist_x = abs(ap_x-x[0])
        dist_y = abs(ap_y-x[1])

        pred_x = b+ m_x*dist_x
        pred_y = b+ m_y*dist_y 
        
        result = (m_x*pred_x + m_y*pred_y)/(m_x+m_y) 
        return min(-10,max(result , -93.0)) 
    
    
    def calculate_zero_mean_offset(self, X, Y):

        clf = Ridge()
        clf.fit(X, Y)         
        self.mean_regressor = clf
        
        self.calculate_mean_offset(X,Y)
        
    
    def predict_zero_mean_offset(self, x):
        return self.mean_regressor.predict(x)        
        
    
    def calculate_mean_offset(self, X, Y):
        
        i_centre = np.argmax(Y)

        b = Y[i_centre]
        ap = X[i_centre,:]  

        self.ap_ksum = np.sum(self.build_vector_covariance(ap))

        X_sorted = X[Y.argsort()]
        Y_sorted = sorted(Y) 
        
        m_x = np.mean([(Y_sorted[j]-b)/abs(ap[0]-X_sorted[j,0]) for j in range(10)] )                       
        m_y = np.mean([(Y_sorted[j]-b)/abs(ap[1]-X_sorted[j,1]) for j in range(10)] )        
  
        self.mean_params = [b, ap[0], ap[1], m_x, m_y]        

        