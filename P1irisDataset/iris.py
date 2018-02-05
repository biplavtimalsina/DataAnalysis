import pandas
#from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#shape
#print(dataset.shape)

#head
#print(dataset.head(20))

#descriptions
#print(dataset.describe())

#class distribution
#print(dataset.groupby('class').size())

#now we will try to visualize the data
#first, using univariate plots
#namely, box and whisker plots

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=True,sharey=True)
#plt.show()

#histograms
#dataset.hist()
#plt.show()

#scatter plot matrix
#scatter_matrix(dataset)
#plt.show()


#Create some models of the data and estimate their accuracy on unseen data

#divide dataset into training and testing dataset
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.20  	#use 20% for test
seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

#lets use different models to classify the dataset
models=[]	
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

scoring='accuracy'

results=[]
names=[]

for name,model in models:
	kfold=model_selection.KFold(n_splits=10,random_state=seed)
	#print(kfold)
	cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
	results.append(cv_results)
	#print(results)
	names.append(name)
	msg="%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
	print(msg)


#Compare Algorithms	
fig=plt.figure()					#creates a empty figure object, is not shown until plt.show() function is called
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)			#controls the size of the inner white subplot where  actual figure is drawn
#print(ax)
plt.boxplot(results)
ax.set_xticklabels(names)
print('names',names)
plt.show()

#testing KNN for validation set
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions=knn.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(">>>>>>>")
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))
