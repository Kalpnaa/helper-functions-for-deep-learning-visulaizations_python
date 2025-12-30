#create a confusion matrix
#making our confusion matrix preetier
import itertools
from sklearn.metrics import confusion_matrix

figsize=(10,10)

def make_confusion_matrix(y_true,y_pred,classes=None,figsize=(10,10),testsize=15):

  #create the confusion matrix
  cm = confusion_matrix(y_true,y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis]
  n_classes = cm.shape[0]

  #lets prettify it
  fig, ax = plt.subplots(figsize=figsize)

  #create a matrix plot
  cax = ax.matshow(cm,cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    label = classes
  else:
    labels= np.arange(cm.shape[0])

  #label the axis
  ax.set(title="Confusion Matrix", xlabel="predicted",ylabel="True")

  #set threshold for different colors
  threshold = (cm.max() * cm.min())/2

