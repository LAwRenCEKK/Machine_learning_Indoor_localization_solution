"""
Author: Jianpeng Liu
Copyrights belong to WiSeR Lab 
Downstream Localiztion models: KNN, DT and RF
"""
import math 
import statistics
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def load_KNN():

    model_name = "K-Nearest Neighbors"
    regr = KNeighborsRegressor(n_neighbors=1, algorithm='kd_tree',
                                leaf_size=50, p=1)
    
    return model_name, regr


def load_Random_Forest():

    model_name = "Random Forest Regressor"
    regr = RandomForestRegressor(n_estimators=100)
    
    return model_name, regr


def load_Decision_Tree():
 
    model_name = "Decision Tree"
    regr = DecisionTreeRegressor()
    
    return model_name, regr



def evaluation_reg(training_data,test_data,l):
    x_train = training_data[:,:5]
    x_test= test_data[:,:5]
    y_train = training_data[:,-2:]
    y_test = test_data[:,-2:]

    data_in =  (x_train, x_test, y_train, y_test)
    model_name, regr = load_KNN()
    error1, prediction1 = run_model_reg(model_name, regr, data_in,l)
    print ("Knn error(m) - "+ str(error1))
    model_name, regr= load_Random_Forest()
    error2,prediction2 = run_model_reg(model_name, regr, data_in,l)
    print ("Random Forest error(m)- "+ str(error2))
    model_name, regr= load_Decision_Tree()
    error3,prediction3 = run_model_reg(model_name, regr, data_in,l)
    print ("Decision Tree error(m)- "+ str(error3))
    print (" ")
    return {"errors":[error1,error2,error3],"results":[prediction1,prediction2,prediction3]}



def run_model_reg(model_name, regr, data, ob):
    x_train, x_test, y_train, y_test = data           
    # Regressor
    fit = regr.fit(x_train, y_train)
    prediction = fit.predict(x_test)
    results = []
    for i in  (y_test-prediction):
        result = math.sqrt(i[0]**2+i[1]**2)
        results.append(result)
    result = statistics.median(results)
    
    return result, prediction


"""
Testing code
"""
# if __name__ == "__main__":

#     files = ["10_labeled_data.csv","propagation_results_10_10.csv",
#         "propagation_results_10_20.csv","propagation_results_10_30.csv"]
#     for f in files:
#         print (str(f))
#         training_data = pd.read_csv(f, header=None)
#         test_data = pd.read_csv("12_test_data.csv",header=None)
#         x_train = training_data.iloc[:,:5].to_numpy()
#         x_test= test_data.iloc[:,:5].to_numpy()
#         y_train = training_data.iloc[:,-2:]
#         y_test = test_data.iloc[:,-2:]

#         data_in =  (x_train, x_test, y_train, y_test)
#         print ("Knn")
#         model_name, regr = load_KNN()
#         error, prediction = run_model(model_name, regr, data_in)
#         print (prediction)
#         print (error)
#         print ("Random_Forest")
#         model_name, regr= load_Random_Forest()
#         error,prediction = run_model(model_name, regr, data_in)
#         print (error)
#         print (prediction)
#         fig, ax = plt.subplot()
#         ax.scatter(prediction[:,0],prediction[:,1])
#         ax.show()
#         print ("Decision_Tree")
#         model_name, regr= load_Decision_Tree()
#         error,prediction = run_model(model_name, regr, data_in)
#         print (prediction)

#         print (error)
#         print (" ")