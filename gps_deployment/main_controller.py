"""
Author: Jianpeng Liu
Copyrights belong to WiSeR Lab 
Downstream Localiztion model GPs
Reference: https://rse-lab.cs.washington.edu/postscripts/gp-localization-rss-06.pdf
"""
import gps_deployment.gaussian as gaussian
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 


class MainController:
    
    def __init__(self, num_ap, training_data):   
        #Close any old plots
        plt.close("all")
        self.number_of_access_points = num_ap   
        self.gaussian_processes = [None]*num_ap
        self.train_data = training_data
        
        
    def load_data(self):
        df = pd.DataFrame(self.train_data)
        if (df.shape[0]<=10):
            df = pd.concat([df,df],axis=0)
        self.wifi_locations = df.iloc[:,self.number_of_access_points:].to_numpy()
        # print (self.wifi_locations)
        self.wifi_rssi = df.iloc[:, :self.number_of_access_points].to_numpy()
        # print (self.wifi_rssi)

        self.wifi_values = np.zeros((len(self.wifi_locations),self.number_of_access_points))
    
    
    def visualize_wifi_data(self, wifi_locations, wifi_values, ap):
        visualize = True
        if(visualize):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_zlim([-100.0,0])
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Wifi Signal Strength(dB)')
            ax.set_title('Wifi measurements for AP # ' + str(ap))
            ax.scatter(wifi_locations[:,0], wifi_locations[:,1], wifi_values, c='orange')
    
    
    def initialize_gaussian_processes(self, visualize=True):
        gaussians = []
        for i in np.arange(0,self.number_of_access_points,1):
            # There are multiple APs, need to create multiple GPs
            # print ("Gaussian Process #" + str(i))
            gp = gaussian.GaussianProcess()
            wifi_rssi_mean_ap = self.wifi_rssi[:,i]
            wifi_rssi_cov_ap = np.zeros(len(wifi_rssi_mean_ap))
            if(visualize == True):
                self.visualize_wifi_data(self.wifi_locations, wifi_rssi_mean_ap,i)
            
            gp.train_gaussian_model_with_params(self.wifi_locations, wifi_rssi_mean_ap, self.default_params())
            gp.set_prior_covariance(wifi_rssi_cov_ap)
            gaussians.append(gp)
            self.wifi_values[:,i] = wifi_rssi_mean_ap 
        self.gaussian_processes = gaussians
    
    #Initialize GPs
    def initialize_gaussian_processes_optimal(self, visualize=True):
        gaussians = []
        for i in np.arange(0,self.number_of_access_points,1):
            # There are multiple APs, need to create multiple GPs
            # print "Gaussian Process #", i
            gp = gaussian.GaussianProcess()
            # wifi_rssi_mean_ap, wifi_rssi_cov_ap = self.preprocess_wifi_data(self.wifi_locations, self.wifi_rssi, i)
            wifi_rssi_mean_ap = self.wifi_rssi[:,i]
            wifi_rssi_cov_ap = np.zeros(53-9)
            if(visualize == True):
                self.visualize_wifi_data(self.wifi_locations, wifi_rssi_mean_ap,i)
            gp.set_param_ranges(i)      
            gp.train_gaussian_model(self.wifi_locations, wifi_rssi_mean_ap)
            gp.set_prior_covariance(wifi_rssi_cov_ap)
            gaussians.append(gp)
            self.wifi_values[:,i] = wifi_rssi_mean_ap            
        self.gaussian_processes = gaussians
        
    
    #Saved parameter rang values    
    def default_params(self):
        return [8,2,0.5]

     
    #Prediction using gaussian, extrapolate for entire grid
    def visualize_gaussian(self, ap_index, gp):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Wifi Signal Strength(dB)')
        ax.set_title('Gaussian fit for AP # ' + str(ap_index))
        constant = 0.01

        # This can be customizd
        ax.set_zlim([-93.0,0])
        X_range = np.arange(0,100,10)
        Y_range = np.arange(0,100,10)
        # print (self.wifi_locations)
        Y_train_rounded = self.wifi_locations
        rlt = []
        Y,X = np.meshgrid(Y_range,X_range)
        Z=np.zeros(shape = X.shape)
        for i in range(0,len(X_range)):
            for j in range(0,len(Y_range)):
                x = X_range[i]
                y = Y_range[j]
                # Z is the result 
                Z[i,j] = gp.predict_gaussian_value([x,y])
                rlt.append(Z[i,j])
                
        ax.plot_surface(X,Y,Z)
        plt.show()
        return rlt 

