#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('heart.csv')


# In[4]:


info = ["age of person","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure","serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)","maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 2 = normal; 1 = fixed defect; 3 = reversable defect"]
for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])


# In[5]:


df.isnull().sum()  #we're calculating the sum of null values --> no null so 0's


# # Taking Care of Duplicates

# In[6]:


data_dup = df.duplicated().any()


# In[7]:


data_dup #contains duplicate value


# In[8]:


df = df.drop_duplicates()


# In[9]:


data_dup = df.duplicated().any()


# In[10]:


data_dup #now duplicate values are removed


# # Data Preprocessing

# 1.Find unique values

# In[11]:


df["cp"].value_counts().count()


# In[12]:


df


# In[13]:


import pandas as pd
unique_number = []
for i in df.columns:
    x = df[i].value_counts().count()
    unique_number.append(x)
pd.DataFrame(unique_number,index=df.columns,columns=["Total Unique Values"])


# Separating categorical and numerical values

# In[14]:


#variables which have larger unique -->numerical else categorical

category_val = [] #categorical columns,qualitative
numerical_val = [] #numerical columns,quantitative

for column in df.columns:
    if df[column].nunique() <=10:
        category_val.append(column)
    else:
        numerical_val.append(column)


# In[15]:


category_val


# In[16]:


numerical_val


# # 3.Exploratory Data Analysis for univariate analysis

# In[17]:


numeric_axis_name=["Age of the Patient","Resting Blood Pressure","Cholestrol","Maximum Heart Rate Achieved","ST Depression"]


# In[18]:


list(zip(numerical_val,numeric_axis_name))


# In[19]:


title_font ={"family":"arial","color":"darkred","weight":"bold","size":15}
axis_font ={"family":"arial","color":"darkblue","weight":"bold","size":13}

for i,z in list(zip(numerical_val,numeric_axis_name)):
    plt.figure(figsize=(8,6), dpi=80)
    sns.distplot(df[i],hist_kws=dict(linewidth = 1, edgecolor ="k"),bins=20)
               
    plt.title(i,fontdict=title_font)
    plt.xlabel(z,fontdict=axis_font)
    plt.ylabel("Density",fontdict=axis_font)
               
    plt.tight_layout()
    plt.show()


# In[20]:


categoric_axis_name = ["Gender","Chest Pain Type","Fasting Blood Sugar","Resting Electrocardiographic Results","Exercise Induced Angina","The Slope of ST Segment","Number of Major Vessels","Thal","Target"]


# In[21]:


list(zip(category_val,categoric_axis_name))


# In[22]:


df["cp"].value_counts()


# In[23]:


list(df["cp"].value_counts())


# In[24]:


list(df["cp"].value_counts().index)


# In[25]:


title_font ={"family":"arial","color":"darkred","weight":"bold","size":15}
axis_font ={"family":"arial","color":"darkblue","weight":"bold","size":13}
for i, z in list(zip(category_val,categoric_axis_name)):
    fig,ax= plt.subplots(figsize=(8,6))
    observation_values=list(df[i].value_counts().index)
    total_observation_values=list(df[i].value_counts())
    ax.pie(total_observation_values,labels=observation_values,autopct='%1.1f%%',startangle=110,labeldistance=1.1)
    ax.axis("equal")
    plt.title((i +"("+ z + ")"), fontdict=title_font)
    plt.legend()
    plt.show()
    
    


# # 4. Exploratory Data Analysis for bivariate analysis - 2 variables 

# 4.1 Numerical var - target (Analysis with FaceGrid)

# In[26]:


numerical_val.append("target")


# In[27]:


numerical_val


# In[28]:


numeric_axis_name = ["Age Of the Patient","Resting Blood Pressure","Cholestrol","Maximum Heart Rate Achieved","ST Depression"]


# In[29]:


list(zip(numerical_val,numeric_axis_name))


# In[30]:


from matplotlib import pyplot as plt
title_font = {"family":"arial","color":"darkred","weight":"bold","size":15}
axis_font = {"family":"arial","color":"darkblue","weight":"bold","size":13}


for i,z in list(zip(numerical_val,numeric_axis_name)):
    graph = sns.FacetGrid(df[numerical_val],hue="target",height = 5,xlim = ((df[i].min() - 10),(df[i].max() + 10)))
    graph.map(sns.kdeplot, i,shade  = True)
    graph.add_legend()
    plt.title(i,fontdict = title_font)
    plt.xlabel(z,fontdict = axis_font)
    plt.ylabel("Density",fontdict = axis_font)
    plt.tight_layout()
    plt.show()


# In[31]:


df[numerical_val].corr()


# In[32]:


df[numerical_val].corr().iloc[:,[-1]]


# Examining missing value

# In[33]:


df[df["thal"] == 0]


# In[34]:


df["thal"] = df["thal"].replace(0,np.nan)


# In[35]:


df.loc[[14,319], :]


# In[36]:


df["thal"].fillna(2, inplace = True)


# In[37]:


df.loc[[14,319], :]


# In[38]:


df["thal"] = pd.to_numeric(df["thal"],downcast = "integer")


# In[39]:


df.loc[[14,319], :]


# In[40]:


#the target is more correlated with talach and oldpeak


# 4.2 Categorical var - target (Analysis with Count Plot)

# In[41]:


category_val


# In[42]:


categoric_axis_name = ["Gender","Chest Pain Type","Fasting Blood Sugar","Resting Electrocardiographic results","Excercise Induced Angina","The Slope Of ST segment","Number Of Major Vessels","Thal","Target"]


# In[43]:


list(zip(category_val,categoric_axis_name))


# In[44]:


from matplotlib import pyplot as plt
title_font = {"family":"arial","color":"darkred","weight":"bold","size":15}
axis_font = {"family":"arial","color":"darkblue","weight":"bold","size":13}



for i,z in list(zip(category_val,categoric_axis_name)):
    plt.figure(figsize = (8,5))
    sns.countplot(x=i, data = df[category_val], hue = "target")
    plt.title(i + " - target", fontdict = title_font)
    plt.xlabel(z, fontdict = axis_font)
    plt.ylabel("Target", fontdict = axis_font)
    plt.tight_layout()
    plt.show()


# In[45]:


df[category_val].corr()


# In[46]:


df[category_val].corr().iloc[:,[-1]] #most correlated is cp & exang


# Examining numeric Variables among themselves (Pair Plot)

# In[47]:


numerical_val.remove("target")


# In[48]:


df[numerical_val].head()


# In[49]:


graph = sns.pairplot(df[numerical_val], diag_kind ="kde")
graph.map_lower(sns.kdeplot,levels = 4, color =".2")
plt.show()


# In[50]:


#numerical values has outliers so use Robust Scaler


# In[51]:


#from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


# In[52]:


st = StandardScaler()
#rb_scaler = RobustScaler()


# In[53]:


scaled_data = st.fit_transform(df[numerical_val])


# In[54]:


scaled_data


# In[55]:


df_scaled = pd.DataFrame(scaled_data,columns = numerical_val)
df_scaled.head()


# # New DF with Melt()

# In[56]:


df_new = pd.concat([df_scaled, df.loc[:, "target"]], axis = 1)


# In[57]:


df_new.head()


# In[58]:


df_new.isnull().sum()


# In[59]:


df1_new = df_new.isnull()


# In[60]:


title_font = {"family":"arial","color":"darkred","weight":"bold","size":15}
axis_font = {"family":"arial","color":"darkblue","weight":"bold","size":13}

for i in list(df1_new):
    fig, ax = plt.subplots(figsize = (8,6))
    
    observation_values = list(df1_new[i].value_counts().index)
    total_values = list(df1_new[i].value_counts())
    
    ax.pie(total_values, labels = observation_values, autopct = "%1.1f%%", startangle = 110, labeldistance = 1.1 )
    ax.axis("equal")
    
    plt.title(i, fontdict = title_font)
    plt.legend()
    plt.show()


# In[61]:


for i in list(df1_new):
    df_new[i].fillna(0,inplace=True)


# In[62]:


df_new.isnull().sum()


# In[63]:


melted_data = pd.melt(df_new, id_vars = "target", var_name = "variables", value_name ="values")


# In[64]:


melted_data


# In[65]:


melted_data.isnull().sum()


# In[66]:


melted_data


# In[67]:


plt.figure(figsize = (8,5))
sns.swarmplot(x = "variables", y = "values", hue ="target", data = melted_data)
plt.show()


# In[68]:


sns.boxplot(x="variables", y="values", hue="target",
                data=melted_data, palette="Set3")


# In[69]:


axis_font = {"family":"arial","color":"darkblue","weight":"bold","size":13}
for i in df[category_val]:
    df_new = pd.concat([df_scaled, df.loc[:, i]],axis = 1)
    melted_data = pd.melt(df_new, id_vars = i, var_name = "variables", value_name = "values")
    
    plt.figure(figsize = (8,5))
    sns.boxplot(x = "variables", y ="values", hue=i , data=melted_data)
    
    plt.xlabel("variables", fontdict = axis_font)
    plt.ylabel("values", fontdict = axis_font)
    plt.tight_layout()
    plt.show()


# In[70]:


df_new2 = pd.concat([df_scaled, df[category_val]], axis = 1)


# In[71]:


df_new2


# In[72]:


df_new_null = df_new2.isnull()
df_new_null


# In[73]:


title_font = {"family":"arial","color":"darkred","weight":"bold","size":15}
axis_font = {"family":"arial","color":"darkblue","weight":"bold","size":13}

for i in list(df_new_null):
    fig, ax = plt.subplots(figsize = (8,6))
    
    observation_values = list(df_new_null[i].value_counts().index)
    total_values = list(df_new_null[i].value_counts())
    
    ax.pie(total_values, labels = observation_values, autopct = "%1.1f%%", startangle = 110, labeldistance = 1.1 )
    ax.axis("equal")
    
    plt.title(i, fontdict = title_font)
    plt.legend()
    plt.show()


# In[74]:


for i in list(df_new_null):
    df_new2[i].fillna(0,inplace=True)


# In[75]:


df_new2.isnull().sum()


# In[76]:


df_new2


# In[77]:


df_new2.corr()


# In[78]:


plt.figure(figsize = (15,10))
sns.heatmap(data = df_new2.corr(),cmap = "Spectral", annot = True, linewidths = 0.5)


# In[79]:


#fbs,ca -> cat , chol,trestbps ->numerical


# In[80]:


numerical_val


# In[81]:


category_val


# Dropping columns with less correlation

# In[82]:


#df.drop(['oldpeak','fbs','ca'],axis = 1,inplace = True)
df.drop(['fbs','ca'],axis = 1,inplace = True)


# In[83]:


df.head()


# Checking Outliers

# In[84]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize = (20,6))
ax1.boxplot(df["age"])
ax1.set_title("age")


ax2.boxplot(df["trestbps"])
ax2.set_title("trestbps")

ax3.boxplot(df["chol"])
ax3.set_title("chol")

ax4.boxplot(df["thalach"])
ax4.set_title("thalach")

plt.show()


# In[85]:


#chol,trestbps more outliers


# In[86]:


#Dealing with Outliers -> trestbps


# In[87]:


from scipy import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize


# In[88]:


z_scores_trtbps = zscore(df["trestbps"])
for threshold in range(1,4):
    print("Threshold value: {}".format(threshold))
    print("Number Of Outliers: {}".format(len(np.where(z_scores_trtbps > threshold)[0])))
    print("------------------------")


# In[89]:


# 51 outliers above 1st threshold
# 13 above 2nd
# 2 above 3rd


# In[90]:


df[z_scores_trtbps >2][["trestbps"]]


# In[91]:


#value assigning to outliers


# In[92]:


#winzerize limit top & bottom


# In[93]:


df[z_scores_trtbps > 2].trestbps.min()


# In[94]:


df[df["trestbps"] < 170].trestbps.max()


# In[95]:


#percent from bottom,top


# In[96]:


winsorise_percent_trtbps = (stats.percentileofscore(df["trestbps"],165)) / 100
print(winsorise_percent_trtbps)


# In[97]:


1 - winsorise_percent_trtbps #upper


# In[98]:


trtbps_win =winsorize(df.trestbps,(0,(1 - winsorise_percent_trtbps)))


# In[99]:


plt.boxplot(trtbps_win)
plt.xlabel("trestbps winsorize",color = "b")


# In[100]:


df["trestbps_winsorize"] = trtbps_win


# In[101]:


df.head()


# Thalach Variable

# In[102]:


#inter quartile range - 50% data 75-25 1.5 more and less
def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5*diff)
    upper_v = q3 + (1.5*diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]
    
    


# In[103]:


thalach_out = iqr(df, "thalach")


# In[104]:


thalach_out #only one outlier


# In[105]:


df.drop([267], axis = 0, inplace = True)


# In[106]:


df["thalach"][270:275]


# In[107]:


plt.boxplot(df["thalach"])


# Oldepeak variable

# In[108]:


#inter quartile range - 50% data 75-25 1.5 more and less
def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5*diff)
    upper_v = q3 + (1.5*diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]
    
    


# In[109]:


oldpeak_out = iqr(df, "oldpeak")


# In[110]:


oldpeak_out


# In[111]:


#apply winsizer 


# In[112]:


#394 low
df[df["oldpeak"] < 4.2].oldpeak.max()


# In[113]:


#find percentile
wins_perc_oldpeak = (stats.percentileofscore(df["oldpeak"], 4.0))/100
print(wins_perc_oldpeak)


# In[114]:


oldpeak_wins = winsorize(df.oldpeak, (0, (1-wins_perc_oldpeak))) #bott,upp


# In[115]:


plt.boxplot(oldpeak_wins)
plt.xlabel("oldpeak_winsorize",color ="b")


# In[116]:


df["oldpeak_winsorize"] =  oldpeak_wins


# In[117]:


df.head()


# cholestral Variable

# In[118]:


#inter quartile range - 50% data 75-25 1.5 more and less
def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5*diff)
    upper_v = q3 + (1.5*diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]


# In[119]:


iqr(df, "chol")


# In[120]:


#apply winsizer - 5 val


# In[121]:


#394 low
df[df["chol"] < 394].chol.max()


# In[122]:


#find percentile
wins_perc_chol = (stats.percentileofscore(df["chol"], 360))/100
print(wins_perc_chol)


# In[123]:


chol_wins = winsorize(df.chol, (0, (1-wins_perc_chol))) #bott,upp


# In[124]:


plt.boxplot(chol_wins)
plt.xlabel("cholestrol_winsorize",color ="b")


# In[125]:


df["cholestrol_winsorize"] =  chol_wins


# In[126]:


df.head()


# In[127]:


df.drop(["trestbps", "chol","oldpeak"], axis = 1, inplace = True)


# In[128]:


df.head()


# Determining Distributions Of Numeric Variables

# In[129]:


fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(1,5, figsize = (20,6))

ax1.hist(df["age"])
ax1.set_title("age")

ax2.hist(df["trestbps_winsorize"])
ax2.set_title("trestbps_winsorize")

ax3.hist(df["thalach"])
ax3.set_title("thalach")

ax4.hist(df["cholestrol_winsorize"])
ax4.set_title("cholestrol_winsorize")


ax5.hist(df["oldpeak_winsorize"])
ax5.set_title("oldpeak_winsorize")

plt.show()


# In[130]:


df[["age", "trestbps_winsorize", "thalach", "cholestrol_winsorize","oldpeak_winsorize"]].agg(["skew"]).transpose()


# Transformation Operations on Unsymmetrical Data

# In[131]:


df["old_win_log"] = np.log(df["oldpeak_winsorize"])
df["old_win_sqrt"] = np.sqrt(df["oldpeak_winsorize"])


# In[132]:


df.head()


# In[133]:


df[["oldpeak_winsorize","old_win_log","old_win_sqrt"]].agg(["skew"]).transpose()


# In[134]:


df.drop(["oldpeak_winsorize","old_win_log"],axis = 1, inplace = True)


# In[135]:


df.head()


# In[136]:


#-0.5 to 0.5 -> symmetrical
#-0.5 to -1.0 -> moderate skew
# > -1.0 , 1.0 -> very skewed
# completion of numerical values


# Applying One Hot Encoding Method to Categorical values

# In[137]:


df_copy = df.copy()


# In[138]:


df_copy.head()


# In[139]:


category_val


# In[140]:


category_val.remove("fbs")
category_val.remove("ca")


# In[141]:


category_val


# In[142]:


df_copy = pd.get_dummies(df_copy, columns = category_val[:-1], drop_first = True)


# In[143]:


df_copy.head()


# Feature Scaling with Robut Scaler

# In[144]:


numerical_val


# In[145]:


new_numeric_val = ["age","thalach","trestbps_winsorize","cholestrol_winsorize"]


# In[146]:


#robust_scaler = RobustScaler()
st = StandardScaler()


# In[147]:


df_copy[new_numeric_val] = st.fit_transform(df_copy[new_numeric_val])


# In[148]:


df_copy.head()


# Separate Data into Test and Training

# In[149]:


from sklearn.model_selection import train_test_split


# In[150]:


x = df_copy.drop(["target"],axis = 1)
y = df_copy[["target"]]


# In[151]:


X_train,X_test,y_train,y_test = train_test_split(x, y , test_size = 0.2, random_state =3)


# In[152]:


X_train.head()


# In[153]:


y_train.head()


# In[154]:


print(f"X_train: {X_train.shape[0]}")
print(f"X_test: {X_test.shape[0]}")
print(f"y_train: {y_train.shape[0]}")
print(f"y_test: {y_test.shape[0]}")


# In[155]:


y_test.isnull().sum()


# # Modeling

# Logistic Regression

# In[156]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[157]:


log_reg = LogisticRegression()
log_reg


# In[158]:


log_reg.fit(X_train,y_train.values.ravel())


# In[159]:


X_test


# In[160]:


y_pred_log = log_reg.predict(X_test)


# In[161]:


y_pred_log


# In[162]:


log_acc= accuracy_score(y_test,y_pred_log)
print("Test Accuracy: {}".format(log_acc))


# K - Fold Cross Validation

# In[163]:


from sklearn.model_selection import cross_val_score


# In[164]:


scores = cross_val_score(log_reg, X_test, y_test.values.ravel(), cv = 7)


# In[165]:


print("Cross-Validation Accuracy Score: ", scores.mean())


# In[166]:


from sklearn.metrics import plot_roc_curve


# In[167]:


#import sklearn.metric.*
plot_roc_curve(log_reg,X_test,y_test,name="Logistic Regression")
plt.title("Logistic Regression Roc Curve and AUC")
plt.plot([0,1],[0,1],"r--")
plt.show()


# In[168]:


from sklearn.model_selection import GridSearchCV


# In[169]:


#log_reg_new=LogisticRegression(C=1, penalty='none',solver='newton-cg', max_iter=100, random_state=1, multi_class='auto')
#log_reg_new
#C=0.01, penalty='l1',solver='liblinear'


# In[170]:


from sklearn import linear_model
C = np.logspace(0, 4, num=10)
penalty = ['l1', 'l2']
solver = ['liblinear', 'saga']
hyperparameters = dict(C=C, penalty=penalty, solver=solver)
logistic = linear_model.LogisticRegression(max_iter = 4000)
gridsearch = GridSearchCV(logistic, hyperparameters)
best_model_grid = gridsearch.fit(X_train,y_train.values.ravel())
print(best_model_grid.best_estimator_)


# In[171]:


logistic.fit(X_train,y_train.values.ravel())


# In[172]:


y_pred = logistic.predict(X_test)


# In[173]:


print("Accuracy score after hyper parameter: {}".format(accuracy_score(y_test,y_pred)))


# In[174]:


from sklearn.ensemble import RandomForestClassifier


# In[175]:


rf = RandomForestClassifier(random_state=5)


# In[176]:


rf.fit(X_train,y_train.values.ravel())


# In[177]:


y_pred5= rf.predict(X_test)
accuracy_score(y_test,y_pred5)


# In[178]:


scores1 = cross_val_score(rf, X_test, y_test.values.ravel(), cv = 7)


# In[179]:


print("Cross-Validation Accuracy Score: ", scores1.mean())


# In[180]:


#pip install -U scikit-learn


# In[181]:


random_forest_new = RandomForestClassifier(random_state = 5)
random_forest_new


# In[182]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid =  {"n_estimators" : [50, 100, 150, 200], 
              "criterion" : ["gini", "entropy"], 
              'max_features': ['sqrt', 'log2'], 
              'bootstrap': [True, False]
              }


# In[183]:


random_forest_grid = GridSearchCV(random_forest_new, param_grid = param_grid)


# In[184]:


random_forest_grid.fit(X_train, y_train.values.ravel())


# In[185]:


print("Best Parameters:", random_forest_grid.best_params_)


# In[186]:


random_forest_new2 = RandomForestClassifier(bootstrap = True, criterion = "gini", max_features = "sqrt", n_estimators = 200, random_state = 5)


# In[187]:


random_forest_new2.fit(X_train, y_train.values.ravel())


# In[188]:


y_pred_ran = random_forest_new2.predict(X_test)


# In[189]:


print("The test accuracy score of Random Forest after hyper-parameter tuning is:", accuracy_score(y_test, y_pred_ran))


# In[190]:


#import sklearn.metric.*
plot_roc_curve(random_forest_new2,X_test,y_test,name="Random forest")
plt.title("Random forest Roc Curve and AUC")
plt.plot([0,1],[0,1],"r--")
plt.show()


# SVM

# In[191]:


from sklearn import svm


# In[192]:


svm = svm.SVC()


# In[193]:


svm.fit(X_train,y_train.values.ravel())


# In[194]:


y_pred2= svm.predict(X_test)
accuracy_score(y_test,y_pred2)


# In[195]:


#import sklearn.metric.*
plot_roc_curve(svm,X_test,y_test,name="SVM")
plt.title("SVM Roc Curve and AUC")
plt.plot([0,1],[0,1],"r--")
plt.show()


# In[196]:


from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(svm, X_test, y_test.values.ravel(), cv = 7)


# In[197]:


print("Cross-Validation Accuracy Score: ", scores2.mean())


# In[198]:


parameters = {'kernel':('linear', 'rbf'), 'C':[1.0, 10.0, 100.0, 1000.0],
              'gamma':[1,0.1,0.01]}


# In[199]:


from sklearn import svm
model = svm.SVC()
clf = GridSearchCV(model, parameters, verbose=2)


# In[200]:


clf.fit(X_train,y_train.values.ravel())


# In[201]:


svc_best_param = clf.best_params_
print("Best params for SVM:", svc_best_param)


# In[202]:


y_pred_sv = clf.predict(X_test)


# In[203]:


print("Accuracy score after hyper parameter: {}".format(accuracy_score(y_test,y_pred_sv)))


# In[ ]:





# KNN

# In[204]:


from sklearn.neighbors import KNeighborsClassifier


# In[205]:


knn = KNeighborsClassifier()
knn.fit(X_train,y_train.values.ravel())
y_pred3 = knn.predict(X_test)
accuracy_score(y_test, y_pred3)


# In[206]:


score=[]
for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train.values.ravel())
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test,y_pred))


# In[207]:


score


# In[208]:


knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train.values.ravel())
y_pred_knn=knn.predict(X_test)
accuracy_score(y_test,y_pred_knn)


# In[209]:


#import sklearn.metric.*
plot_roc_curve(knn,X_test,y_test,name="KNN")
plt.title("KNN Roc Curve and AUC")
plt.plot([0,1],[0,1],"r--")
plt.show()


# In[210]:


grid_params = { 'n_neighbors' : [5,7,8,15,16,20],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}


# In[211]:


gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=5, n_jobs = -1)


# In[212]:


g_res = gs.fit(X_train, y_train.values.ravel())
g_res.best_score_
g_res.best_params_


# In[213]:


knn = KNeighborsClassifier(n_neighbors = 7, weights = 'uniform',algorithm = 'brute',metric = 'manhattan')
knn.fit(X_train, y_train.values.ravel())


# In[214]:


y_pred_ran = g_res.predict(X_test)


# In[215]:


print("Accuracy score after hyper parameter: {}".format(accuracy_score(y_test,y_pred_ran)))


# In[216]:


#y_hat = knn.predict(X_train)
#y_knn = knn.predict(X_test)


# In[217]:


#from sklearn.metrics import accuracy_score
#print('Training set accuracy: ',accuracy_score(y_train, y_hat))
#print('Test set accuracy: ',accuracy_score(y_test, y_knn))


# Gradient Boosting Algorithm

# In[218]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train.values.ravel())


# In[219]:


ypred_gradient = gbc.predict(X_test)
accuracy_score(y_test,ypred_gradient)


# In[220]:


from sklearn.model_selection import cross_val_score
scores_gbc = cross_val_score(gbc, X_test, y_test.values.ravel(), cv = 7)
print("Cross-Validation Accuracy Score: ", scores_gbc.mean())


# In[221]:


plot_roc_curve(gbc,X_test,y_test,name="GBC")
plt.title("Gradient BoostingRoc Curve and AUC")
plt.plot([0,1],[0,1],"r--")
plt.show()


# In[222]:


from sklearn.tree import DecisionTreeClassifier


# In[223]:


dec_tree = DecisionTreeClassifier(random_state = 5)


# In[224]:


dec_tree.fit(X_train, y_train)


# In[225]:


y_pred_dec = dec_tree.predict(X_test)


# In[226]:


print("The test accuracy score of Decision Tree is:", accuracy_score(y_test, y_pred_dec))


# In[227]:


scores = cross_val_score(dec_tree, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())


# In[228]:


final_data=pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                           'ACC':[accuracy_score(y_test,y_pred_log),
                                 accuracy_score(y_test,y_pred_sv),
                                 accuracy_score(y_test,y_pred_knn),
                                 accuracy_score(y_test,y_pred_dec),
                                 accuracy_score(y_test,y_pred_ran),
                                 accuracy_score(y_test,ypred_gradient )]})


# In[229]:


final_data


# In[230]:


import seaborn as sns


# In[231]:


sns.barplot(final_data['Models'],final_data['ACC'])


# Prediction on new data

# In[277]:


df.head()


# In[279]:


df_copy2 = df.copy()
df_copy2


# In[280]:


from sklearn.model_selection import train_test_split
x1 = df_copy2.drop(["target"],axis = 1)
y1 = df_copy2[["target"]]


# In[281]:


X_train1,X_test1,y_train1,y_test1 = train_test_split(x1, y1 , test_size = 0.2, random_state =3)


# In[282]:


X_train1.head()


# In[283]:


y_train1.head()


# save model using joblib

# In[290]:


log_reg2 = LogisticRegression()
log_reg2.fit(X_train1,y_train1.values.ravel())


# In[ ]:





# In[291]:


import joblib
joblib.dump(log_reg2,'model_joblib_heart')
model=joblib.load('model_joblib_heart')
#model.predict(new_data)


# In[297]:


#inp_data = (57,1,2,0,150,0,1,3,128,229,1)
inp_data = (45,0,1,1,138,0,1,2,112,160,0)
inp_data_numpy = np.asarray(inp_data)
inp_reshape = inp_data_numpy.reshape(1,-1)

predict = model.predict(inp_reshape)
print(predict)


# In[298]:


if predict[0] == 0:
    print("No Disease")
else:
    print("Have Heart Disease")


# In[ ]:




