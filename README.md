# CS989-Project-code
CS989 Project code

#import the various neccessary packages
import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

# Read in my dataset and call it golf
golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv", header=0)

# Gain an idea of what my dataset looks like
golf.head()
golf.info()
golf.describe()

# Looks at summary stats for the column NUMBER_OF_WINS
grp_by_win = golf.groupby("NUMBER_OF_WINS")
grp_by_win.mean()
# There is one NA value in POINTS_BEHIND_LEAD, because one player is in the lead, we should make the NA value = 0
golf = golf.fillna(0)

# Plot a histogram to analyse what the average carry distance (Distance from tee to the point of ground impact on Par 4 and Par 5 tee shots where a valid radar measurement was     # taken)
Avg_Carry = golf["AVG_CARRY_DISTANCE"]
num_bins=25
plt.hist(Avg_Carry, num_bins, facecolor="green", alpha=0.5)
plt.xlim([245, 305])
plt.xticks(np.arange(250, 310, 5))
plt.xlabel("Average Carry distance of Drive")
plt.ylabel("Count")
plt.show()
# REMOVED ALL UNWANTED COLUMN VARIABLES
golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

# Generate a correlation matrix to see which variables show a relationship
golf.corr()

# Create heatmap
corr = golf.corr()
sns.heatmap(corr)
plt.show()
# Make boxplot of Strokes Gained variables
Strokes_Gained.plot(kind="box", subplots=True, layout=(2,2))
plt.show()

#Heatmap of Strokes Gained with winning variables
corr = SG_Wins.corr()
sns.heatmap(corr)
plt.show()

x = golf["AVG_CARRY_DISTANCE"]
y = golf["POINTS"]
scipy.stats.pearsonr(x, y)
import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.values[:,1:69]
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

# unsupervised - hierarchical methods - wins
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_wins))
Target_Win_encoded = LabelEncoder().fit_transform(target_wins)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
            continue
        else:
            print(a,l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_data)
            print("silhouette score =", metrics.silhouette_score(scaled_data, model.labels_))
            print("completeness score =", metrics.completeness_score(Target_Win_encoded, model.labels_))
            print("homogeneity score =",metrics.homogeneity_score(Target_Win_encoded, model.labels_))

# K-means - wins
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_wins))
Target_Win_encoded = LabelEncoder().fit_transform(target_wins)
for k in range(2,8):
 kmeans = cluster.KMeans(n_clusters=k)
 kmeans.fit(scaled_data)
 print(k)
 print("silhouette_score =", metrics.silhouette_score(scaled_data, kmeans.labels_))
 print("completeness_score =", metrics.completeness_score(Target_Win_encoded, kmeans.labels_))
 print("homogeneity_score =", metrics.homogeneity_score(Target_Win_encoded, kmeans.labels_))

# unsupervised - hierarchical methods - top 10's
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_t10))
Target_T10_encoded = LabelEncoder().fit_transform(target_t10)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
            continue
        else:
            print(a,l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_data)
            print("silhouette score =", metrics.silhouette_score(scaled_data, model.labels_))
            print("completeness score =", metrics.completeness_score(Target_T10_encoded, model.labels_))
            print("homogeneity score =",metrics.homogeneity_score(Target_T10_encoded, model.labels_))

# k-means - top10
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_t10))
Target_T10_encoded = LabelEncoder().fit_transform(target_t10)
for k in range(2, 20):
 kmeans = cluster.KMeans(n_clusters=k)
 kmeans.fit(scaled_data)
 print(k)
 print("silhouette score =", metrics.silhouette_score(scaled_data, kmeans.labels_))
 print("completeness score =", metrics.completeness_score(Target_T10_encoded, kmeans.labels_))
 print("homogeneity score =", metrics.homogeneity_score(Target_T10_encoded, kmeans.labels_))

# unsupervised - hierarchical methods - points
n_samples, n_features = scaled_data.shape
n_digits = 5
Target_Points_encoded = LabelEncoder().fit_transform(target_points)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
            continue
        else:
            print(a,l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_data)
            print(metrics.silhouette_score(scaled_data, model.labels_))
            print(metrics.completeness_score(Target_Points_encoded, model.labels_))
            print(metrics.homogeneity_score(Target_Points_encoded, model.labels_))

#K-means - Points
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_points))
Target_Points_encoded = LabelEncoder().fit_transform(target_points)
for k in range(2, 20):
 kmeans = cluster.KMeans(n_clusters=k)
 kmeans.fit(scaled_data)
 print(k)
 print(metrics.silhouette_score(scaled_data, kmeans.labels_))
 print(metrics.completeness_score(Target_Points_encoded, kmeans.labels_))
 print(metrics.homogeneity_score(Target_Points_encoded, kmeans.labels_))


import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.loc[:,("SG_PUTTING_PER_ROUND", "SG:ARG","SG:APR", "SG:OTT")].values

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

# unsupervised - hierarchical methods - wins
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_wins))
Target_Win_encoded = LabelEncoder().fit_transform(target_wins)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
            continue
        else:
            print(a,l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_data)
            print("silhouette score =", metrics.silhouette_score(scaled_data, model.labels_))
            print("completeness score =", metrics.completeness_score(Target_Win_encoded, model.labels_))
            print("homogeneity score =",metrics.homogeneity_score(Target_Win_encoded, model.labels_))

# K-means - wins
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_wins))
Target_Win_encoded = LabelEncoder().fit_transform(target_wins)
for k in range(2,8):
 kmeans = cluster.KMeans(n_clusters=k)
 kmeans.fit(scaled_data)
 print(k)
 print("silhouette_score =", metrics.silhouette_score(scaled_data, kmeans.labels_))
 print("completeness_score =", metrics.completeness_score(Target_Win_encoded, kmeans.labels_))
 print("homogeneity_score =", metrics.homogeneity_score(Target_Win_encoded, kmeans.labels_))

# unsupervised - hierarchical methods - top 10's
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_t10))
Target_T10_encoded = LabelEncoder().fit_transform(target_t10)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
            continue
        else:
            print(a,l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_data)
            print("silhouette score =", metrics.silhouette_score(scaled_data, model.labels_))
            print("completeness score =", metrics.completeness_score(Target_T10_encoded, model.labels_))
            print("homogeneity score =",metrics.homogeneity_score(Target_T10_encoded, model.labels_))

# k-means - top10
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_t10))
Target_T10_encoded = LabelEncoder().fit_transform(target_t10)
for k in range(2, 20):
 kmeans = cluster.KMeans(n_clusters=k)
 kmeans.fit(scaled_data)
 print(k)
 print("silhouette score =", metrics.silhouette_score(scaled_data, kmeans.labels_))
 print("completeness score =", metrics.completeness_score(Target_T10_encoded, kmeans.labels_))
 print("homogeneity score =", metrics.homogeneity_score(Target_T10_encoded, kmeans.labels_))

# unsupervised - hierarchical methods - points
n_samples, n_features = scaled_data.shape
n_digits = 10
Target_Points_encoded = LabelEncoder().fit_transform(target_points)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
            continue
        else:
            print(a,l)
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_data)
            print(metrics.silhouette_score(scaled_data, model.labels_))
            print(metrics.completeness_score(Target_Points_encoded, model.labels_))
            print(metrics.homogeneity_score(Target_Points_encoded, model.labels_))

#K-means - Points
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target_points))
Target_Points_encoded = LabelEncoder().fit_transform(target_points)
for k in range(2, 20):
 kmeans = cluster.KMeans(n_clusters=k)
 kmeans.fit(scaled_data)
 print(k)
 print(metrics.silhouette_score(scaled_data, kmeans.labels_))
 print(metrics.completeness_score(Target_Points_encoded, kmeans.labels_))
 print(metrics.homogeneity_score(Target_Points_encoded, kmeans.labels_))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
scores = ["Silhouette","Completeness","Homogeneity" ]
values = [0.2473655747958717, 0.9743591774485625, 0.464525227184008]
ax.bar(scores,values)
plt.ylabel("Value of Score")
plt.title("Result of Scores using 16 clusters in K-Means")
plt.show()
import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.values[:,1:69]
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,target_points, test_size = 0.30)

Y_test = np.array(Y_test,dtype=float)
Y_train = np.array(Y_train,dtype=float)
X_test = np.array(X_test,dtype=float)
X_train = np.array(X_train,dtype=float)

print("LOGISTIC REGRESSION")
print("**************************************")
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\nDecision Tree")
print("**************************************")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\n KNN")
print("**************************************")
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("\n\n Naive Bayes")
print("**************************************")
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.values[:,1:69]
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,target_t10, test_size = 0.30)

Y_test = np.array(Y_test,dtype=float)
Y_train = np.array(Y_train,dtype=float)
X_test = np.array(X_test,dtype=float)
X_train = np.array(X_train,dtype=float)

print(np.unique(target_t10))

print("LOGISTIC REGRESSION")
print("**************************************")
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\nDecision Tree")
print("**************************************")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\n KNN")
print("**************************************")
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("\n\n Naive Bayes")
print("**************************************")
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.values[:,1:69]
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 
X = np.delete(X, 1, 1) 

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,target_wins , test_size = 0.30)

Y_test = np.array(Y_test,dtype=float)
Y_train = np.array(Y_train,dtype=float)
X_test = np.array(X_test,dtype=float)
X_train = np.array(X_train,dtype=float)

print("LOGISTIC REGRESSION")
print("**************************************")
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\nDecision Tree")
print("**************************************")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\n KNN")
print("**************************************")
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("\n\n Naive Bayes")
print("**************************************")
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.loc[:,("SG_PUTTING_PER_ROUND", "SG:ARG","SG:APR", "SG:OTT")].values

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,target_points, test_size = 0.30)

Y_test = np.array(Y_test,dtype=float)
Y_train = np.array(Y_train,dtype=float)
X_test = np.array(X_test,dtype=float)
X_train = np.array(X_train,dtype=float)

print("LOGISTIC REGRESSION")
print("**************************************")
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\nDecision Tree")
print("**************************************")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\n KNN")
print("**************************************")
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("\n\n Naive Bayes")
print("**************************************")
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.loc[:,("SG_PUTTING_PER_ROUND", "SG:ARG","SG:APR", "SG:OTT")].values

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,target_t10, test_size = 0.30)

Y_test = np.array(Y_test,dtype=float)
Y_train = np.array(Y_train,dtype=float)
X_test = np.array(X_test,dtype=float)
X_train = np.array(X_train,dtype=float)

print("LOGISTIC REGRESSION")
print("**************************************")
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\nDecision Tree")
print("**************************************")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\n KNN")
print("**************************************")
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("\n\n Naive Bayes")
print("**************************************")
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
import seaborn as sns

golf =  pd.read_csv("C:/Users/44788/Documents/MSc Data Analytics/CS989/golf_2017.csv")

golf = golf.fillna(0)

golf = golf.drop(columns=["BOGEYS_MADE","TOTAL_O/U_PAR","FASTEST_CH_SPEED", "TOTAL_3_PUTTS", "SLOWEST_CH_SPEED", 
                          "TOTAL_DRIVES_FOR_320+", "TOTAL_DRIVES", "FASTEST_BALL_SPEED", "TOTAL_ROUGH", "SLOWEST_BALL_SPEED",
                          "TOTAL_FAIRWAY_BUNKERS", "TOTAL_STROKES", "HIGHEST_SF", "LOWEST_SF", "LOWEST_LAUNCH_ANGLE", 
                          "STEEPEST_LAUNCH_ANGLE", "HIGHEST_SPIN_RATE","LOWEST_SPIN_RATE", "LONGEST_ACT.HANG_TIME",
                          "SHORTEST_ACT.HANG_TIME", "LONGEST_CARRY_DISTANCE", "SHORTEST_CARRY_DISTANCE", 
                          "NUMBER_OF_SAVES","NUMBER_OF_BUNKERS","PAR_OR_BETTER","MISSED_GIR","FAIRWAYS_HIT","POSSIBLE_FAIRWAYS",
                          "ATTEMPTS_GFG","NON-ATTEMPTS_GFG","RTP-GOING_FOR_THE_GREEN", "RTP-NOT_GOING_FOR_THE_GRN","HOLE_OUTS",
                          "POINTS_BEHIND_LEAD", "HOLES_PLAYED","TOTAL_ROUNDS","ROUNDS_PLAYED","MEASURED_ROUNDS", "TOTAL_SG:PUTTING"])

#Will use 3 different variables to test clustering on, will predict wins will be least effective 
target_wins = golf.values[:,3]
target_t10 = golf.values[:,4]
target_points = golf.values[:,2]

X = golf.loc[:,("SG_PUTTING_PER_ROUND", "SG:ARG","SG:APR", "SG:OTT")].values

# Scale the data that we are going to use for clustering
scaled_data = scale(X)

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,target_wins , test_size = 0.30)

Y_test = np.array(Y_test,dtype=float)
Y_train = np.array(Y_train,dtype=float)
X_test = np.array(X_test,dtype=float)
X_train = np.array(X_train,dtype=float)

print("LOGISTIC REGRESSION")
print("**************************************")
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\nDecision Tree")
print("**************************************")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

print("\n\n KNN")
print("**************************************")
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))


print("\n\n Naive Bayes")
print("**************************************")
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
print(model)
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))
