# Machine_learning_Indoor_localization_solution

Conventional fingerprint-based localization solutions requires site surveys at targets areas which are labour-extensive and time-consuming. The proposed solution leverages crowdsensing and semi-supervised learning techniques to reduce labour and time cost in conducting site surveys effectively.  

# Crowd-sensing Techniques 

A passive data acquisition feature is developed and integrated into **MacQuest** mobile app allowing to collect unlabeled WiFi fingerprints from users passively.

## *WiFiDataCollection* Package
***WiFiDataCollection*** package has been developed using **Android Studio** in Java. The package  is independent to the existing code base of **MacQuest**. The package includes six objects andit achieves the desired functionalities with only a minor modification to the existingcode base. The package includes the objects as follows,

 - ***Config***: Config object contains Http request utility functions which are usedto set up the request contents. 
   
 - ***FileUploader***: FileUploader object is used to upload file to the back-endserver.
 -  ***MiscHelper***: MiscHelper is the object used to parse the back-end http re-sponse.
 - ***PermissionRequest***: PermissionRequest is the object used to request thepermission on the run which is required for Android 6.0 (SDK 23)
 - ***WiFiDataManger***: WiFiDataManger is the major object used to control the workflow such as start scanning, stop scanning and pack the collected data

## Usage of *WiFiDataCollection* Package

Instead of conducting WiFi scan-ning constantly in background causing battery drain issues, an opportunistic data collection method is adopted. Whenever a user uses the **MacQuest** app, passive data collection will be triggered. It scans anduploads WiFi RSS data every 5 second. Specifically speaking, a ***wifiDataManager*** object will be initialized and WiFi scanning task will be scheduled by calling ***wifiDataManager.startcollection()*** method, when ***OnResumefunction*** is reached (User opens the App). Upon complete each scan, data will bepacked and uploaded to the back-end server. The ***Handler*** package is used to repeatscanning and uploading tasks periodically. The ***wifiDataManager*** object will be de-stroyed and the periodic tasks will be cancelled if ***OnDestroy*** or ***OnPause*** functions reached (User exists the App or user locks the screen).

# Semi-supervised Learning Techniques
*WiFiDataCollection* allows passive data collection of unlabeled WiFi fingerprints. By utilizing semi-supervised learning techniques to infer labels of unlabeled data, we can increase coverage of site surveys at target areas and reduce localization error by a limited number of labeled data. 

## Distance-aware Graph-based Semi-supervised Learning (DG-SSL)
The **DG-SSL** algorithm is developed in Python3 saved at *DG_SSL/dg_ssl.py*. **DG_SSL** algorithm constructs a graph whose nodes are the labeled and unlabeled fingerprints firstly. Using the similar measurement algorithm, the label of unlabeled fingerprint are inferred by the labeled fingerprint. Referring to the study *"Indoor Positioning and Distance-aware Graph-based Semi-supervised Learning Method"*, ***dg_ssl.py*** is developed to implement **DG_SSL** algorithm.

## Localization Performance Evaluation
To evaluate the performance of DG_SSL  algorithm, four downstream localization models K-nearest neighbour(KNN), Random Forest (RF), Decision Tree (DT) and Gaussian Processes (GPs) are used during the experiment. The **localization_models.py** underneath the folder of ***downstream_models*** contains implementations of KNN, RF and DT models. The files of GPs model are under the folder of ***gps_deployment***. Two data-sets are used to evaluate the localization performance, which are from **McMaster Data Collection Campaign (MDCC)** and **UJIIndoorLoc Data Set**.  Since MDCC data are saved in the pbf format inside the folder of **FpData**, data needs to be parsed firstly.  **Utils.py** provides handy functions to parse and manipulate the MDCC data. **UJIIndoorLoc**  data are saved in a csv file called *trainingData.csv*.
Five buildings are selected to deploy the localization models and the experimental processes and results are recored in the notes as follows,

- ***ITB.ipynb***
- ***ETB.ipynb***
- ***IASH.ipynb***
- ***Uji1.ipynb***
- ***Uji2.ipynb***


## Tree of the Project 

├── DG_SSL
│   └── dg_ssl.py
├── downstream_models
│   └── localization_models.py
├── gps_deployment
│   ├── gaussian.py
│   ├── gps_starting.py
│   └── main_controller.py
├── building_dict.json
├── Data_pb2.py
├── Utils.py
├── Metadata_ujiindoorloc.txt
├── Metadata_mdcc.txt
├── trainingData.csv
├── FpData
├── ETB.ipynb
├── IAHS.ipynb
├── ITB.ipynb
├── uji1.ipynb
└── uji2.ipynb
