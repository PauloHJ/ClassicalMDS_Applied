#IMPORTING LIBRARIES & PACKAGES
from os import X_OK
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
from numpy import linalg as la
import matplotlib.pylab as plt

#FUNCTION TO CALCULATE DISTANCE BETWEEN 2 CITIES
def DistLatLon2Km(lat1,lon1,lat2,lon2):
    worldRad = 6371
    distLat = np.deg2rad(lat2 - lat1)
    distLon = np.deg2rad(lon2 - lon1)
    a = np.sin(distLat/2) * np.sin(distLat/2) + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(distLon/2) * np.sin(distLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    d = worldRad * c; # Distance in km
    return d;

#IMPORTING DATASET & EXTRACTING DATA
df = pd.read_csv('C:\Paulo\KTH\Advanced Machine Learning\Dimensional Reduction\MXCities.csv')
fields = len(df. columns) 
cityArr = df.to_numpy()
numOfCities = int(cityArr.size/fields)

#DECLARATION OF ARRAYS & MATRICES
cityDiffM = [[0 for x in range(numOfCities)] for y in range(numOfCities)] 
cityName = [0 for x in range(numOfCities)]

#BUILDING THE DISTANCE DIFFERENCE MATRIX
for j in range(numOfCities):
    cityName[j] = cityArr[j][0]   #EXTRACTING NAME OF THE CITIES
    for i in range(numOfCities):
        cityDiffM[j][i]=DistLatLon2Km(cityArr[j][1],cityArr[j][2],cityArr[i][1],cityArr[i][2])
        cityDiffM[j][i]=cityDiffM[j][i]*cityDiffM[j][i]

#BUILDING MATRIX J FOR DOUBLE CENTERING
J = [[0 for x in range(numOfCities)] for y in range(numOfCities)] 
for j in range(numOfCities):
    for i in range(numOfCities):
        if(i == j):
            J[j][i] = 1 - (1/numOfCities)
        else:
            J[j][i] = -(1/numOfCities)

cityDiffM = np.array(cityDiffM)
J = np.array(J)
JP = J.dot(cityDiffM)
B = -0.5*JP.dot(J) #DOUBLE CENTERED MATRIX

#CALCULATE EIGENVALUES & EIGENVECTORS OF DOUBLE CENTERED MATRIX
V,D = la.eig(B)
D = D.transpose()
E = np.array([np.array(D[0]),np.array(D[1])])
E = E.transpose()
V = np.sqrt(V)
a = np.zeros([2,2], float)
np.fill_diagonal(a,[V[0],V[1]])

#RESULT MATRIX IN 2D
res = E.dot(a)
res = res.transpose()

#PLOT THE OBTAINED DATAPOINTS IN 2D
cat_col = df['State'].astype('category') #CONVERTING INTO CATEGORICAL DATA 
cat_col = cat_col.cat.codes #ENCODING CATEGORICAL DATA INTO COLOR CODES
plt.scatter(res[0], res[1], c=cat_col)

#ANNOTATE THE PRINCIPAL POINTS
#  for i, txt in enumerate(cityName):
#      plt.annotate(txt, (res[0][i], res[1][i]))