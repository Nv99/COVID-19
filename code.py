"""
here we are going to write a machine learning code which will help us to classify the differece in between the people having corona 
virus or not
The dataset for this is downloded from kaggle which includes the different X-Ray images of people showing whether they are suffering from
pnemonia(corona) or are normal(i.e., did not have any issues)

"""
"""
To create this model we will use different libraries of python such as using pandas library for reading Comma Seperated Values file
(csv file) to using computer vision library such as cv2 for reading and resizeing the image etc

"""

import pandas as pd                   #data processing for CSV file
import numpy as np                    #linear alzebra


#the thing inside the brackets is the path where I have saved my csv file 
#the command pd.read_csv will help us to read the contents of the csv data

corona = pd.read_csv('C:/Users/DELL/Downloads/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

corona.head()                        #this command will give us the overview of the csv files contents
                                     
