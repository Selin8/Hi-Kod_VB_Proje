#!/usr/bin/env python
# coding: utf-8

# # Hi-Kod Veri Bilimi Atölyesi Mezuniyet Projesi
# ## COVİD'in Psikolojik Etkilerinin Araştırılması

# Veri Kümesi Hakkında
# İnsanların karantinaya ilişkin görüşlerini anlamak için 2020'de yapılan anketin yanıtları. Veri kümesi Google Formlar yardımıyla toplandı. Veriler, insanlar tarafından sağlanan çeşitli soruların yanıtlarını içerir. Formdaki soruların çoğu, büyük/küçük harfe duyarlı sorunlardan kaçınmak için çoktan seçmeli olarak sunuldu. Bu veri seti bir araştırma makalesi yayınlamak için kullanıldı.
# 

# ## Bu Projede Neler Var?
# 1. [Verinin İçeriği](#1)
# 1. [Kütüphanelerin İndirimesi](#2)
# 1. [Verinin Yüklenmesi ve İncelenmesi](#3)
# 1. [Veri Temizleme ve Ön İşleme](#4)
#    1. Kayıp Değerlerin Tespiti
#    1. Gereksiz Değişkenlerin Çıkarılması
#    1. Değerlerin Kontrolü
# 1. [Veri Görselleştirme](#5)
#    1. Tek Değişen Analizi
#    1. Çoklu Değişen Analizi
# 1. [Modelleme](#6)
#    1. Logistic Regression
#    1. K- Nearest Neighbors
#    1. Decision Tree
#    1. Random Forest
#    1. Support Vector Machine

# <a id="1"></aa>
# ### Verinin İçeriği
# * **age:** Kişinin yaş grubu    
# * **gender:** Kişinin cinsiyeti   
# * **occupation:** Kişinin çalıştığı meslek/sektör   
# * **line_of_work:** Kişinin gerçekleştirdiği iş kolu   
# * **time_bp:** Pandemiden önce işte geçirilen süre  
# * **time_dp:** Pandemi sırasında işte geçirilen süre    
# * **travel_time:** Harcanan seyahat süresi   
# * **easyof_online:** Çevrimiçi çalışmanın derecelendirmesi  
# * **home_env:** Ev ortamını sevme    
# * **prod_inc:** Verimlilik artışı değerlendirmesi   
# * **sleep_bal:** Uyku döngüsünün derecelendirilmesi   
# * **new_skill:** Herhangi bir yeni becerinin öğrenilip öğrenilmediği  
# * **fam_connect:** Kişinin ailesiyle ne kadar iyi bağlantı kurduğunu derecelendirilmesi 
# * **relaxed:** Kişinin ne kadar rahat hissettiğinin derecelendirilmesi  
# * **self_time:** Ne kadar kişisel zaman elde edildiğinin değerlendirilmesi   
# * **like_hw:** Evden çalışmayı sevmek    
# * **dislike_hw:** Evden çalışmayı sevmemek    
# * **prefer:** Kişinin evden/ofisten çalışma tercihi    
# * **certaindays_hw:** Belirli günlerin evden çalışmanın gerekli olup olmadığını beğenmek     
# * **X:** Özel Sütun   
# * **time_bp.1:** Özel Sütun   
# * **travel_new:** Özel Sütun  
# * **net_diff:** Özel Sütun 

# <a id="2"></aa>
# ### Kütüphanelerin İndirimesi 

# In[44]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression


# <a id="3"></aa>
# ### Verinin Yüklenmesi ve İncelenmesi

# In[2]:


data= pd.read_csv("C:/Users/Selin Şahin/Desktop/psyco.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.columns


# In[6]:


data.describe()


# <a id="4"></aa>
# ### Veri Temizleme ve Ön İşleme

# ##### Kayıp Değerlerin Tespiti

# In[7]:


data.isnull().sum()


# ##### Gereksiz Değişkenlerin Çıkarılması

# In[8]:


data= data.drop(['line_of_work','Unnamed: 19','travel+work','time_bp.1'], axis= 1)


# In[9]:


data.head()


# In[10]:


len(data.columns)


# ##### Değerlerin Kontrolü

# In[11]:


data.gender.unique()


# In[12]:


data.age.unique()


# In[13]:


data['age'] = data['age'].replace('Dec-18','0-18')
data['age']


# In[14]:


data['occupation'].unique()


# In[15]:


data['occupation'] = data['occupation'].replace('Medical Professional aiding efforts against COVID-19', 'Medical Professional')


# In[16]:


data['prefer'].unique()


# In[17]:


data['prefer'] = data['prefer'].replace({'Complete Physical Attendance':'Physical', 'Work/study from home':'Remote'})


# In[18]:


data.head()


# <a id="5"></aa>
# ### Veri Görselleştirme

# ##### Tek Değişen Analizi

# In[19]:


data.gender.value_counts()


# In[20]:


data.occupation.value_counts()


# In[21]:


data.age.value_counts()


# In[22]:


data.certaindays_hw.value_counts()


# ##### Çoklu Değişen Analizi

# In[23]:


occupation_time_data = data[['occupation', 'time_bp', 'time_dp']]


# In[24]:


occupation_avg_time = occupation_time_data.groupby('occupation').mean()
occupation_avg_time.plot(kind='bar', figsize=(8, 5), color=['lightgreen','salmon'])
plt.xlabel('Meslekler')
plt.ylabel('Ortalama Çalışma Saatleri')
plt.title('Mesleklere Göre Pandemi Öncesi ve Sırasındaki Ortalama Çalışma Saatleri')
plt.xticks(rotation= 45)
plt.legend(['Pandemiden Önceki Zaman', 'Pandemi Sırasındaki Zaman'])
plt.tight_layout()
plt.show()


# In[25]:


list = ["travel_time","prod_inc","sleep_bal","new_skill","fam_connect","relaxed","self_time"]
sns.boxplot(data.loc[:, list], orient = "v", palette = "Set3")
plt.ylabel('Değişen Aralığı')
plt.title('Boxplot')
plt.xticks(rotation= 45)
plt.show()


# In[26]:


fig= plt.subplots(figsize=(8, 5))
sns.countplot(x='occupation', hue='gender', data=data, palette=['lightblue', 'pink','lightgreen'])
plt.title('Cinsiyete Göre Meslek Dağılımı')
plt.xlabel('Meslek')
plt.ylabel('Kişi Sayısı')
plt.legend(['Erkek', 'Kadın','Belirtmek İstemeyen'])
plt.xticks(rotation= 45)
plt.show()


# In[27]:


fig= plt.subplots(figsize=(8, 4))
sns.countplot(x='prefer', hue='gender', data=data, palette=['lightblue', 'pink','lightgreen'])
plt.title('Cinsiyete Göre Çalışma Şekli Tercihi')
plt.xlabel('Evden/Ofisten Çalışma Şekli Tercihi')
plt.ylabel('Kişi Sayısı')
plt.legend(['Erkek', 'Kadın','Belirtmek İstemeyen'])
plt.show()


# In[28]:


list_numeric= ["time_bp","time_dp","travel_time","prod_inc","sleep_bal","new_skill","fam_connect","relaxed","self_time","like_hw","dislike_hw"]
numeric = data.loc[:, list_numeric]
numeric.head()


# In[29]:


corr= numeric.corr()
mask= np.triu(np.ones_like(corr, dtype=bool))
f,ax= plt.subplots(figsize=(8, 8))
cmap= sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, annot = True, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": .5}, fmt = ".2f")
plt.title('Değişkenlerin Korelasyon Matrisi')
plt.show()


# <a id="6"></aa>
# ### Modelleme

# In[30]:


datac = data.copy()


# In[31]:


new_data = pd.get_dummies(datac, columns = ['age', 'gender','occupation','easeof_online','home_env','certaindays_hw'])
new_data.head()


# In[32]:


new_data.shape


# In[33]:


X = new_data.drop(columns='prefer', axis=1)
y = new_data["prefer"]


# In[34]:


ss = StandardScaler()
X_scaler = ss.fit_transform(X)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size= 0.2, random_state= 0)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# #### Logistic Regression

# In[38]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print( 'Accuracy Score',accuracy_score(y_test, y_pred)*100)
print("\nClassification Report: \n", classification_report(y_test, y_pred))


# #### K- Nearest Neighbors

# In[39]:


knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train, y_train)
y_pred = knn1.predict(X_test)
print( 'Accuracy Score',accuracy_score(y_test, y_pred)*100)
print("\nClassification Report: \n", classification_report(y_test, y_pred))


# #### Decision Tree

# In[40]:


dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print( 'Accuracy Score',accuracy_score(y_test, y_pred)*100)
print("\nClassification Report: \n", classification_report(y_test, y_pred))


# #### Random Forest

# In[41]:


rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print( 'Accuracy Score',accuracy_score(y_test, y_pred)*100)
print("\nClassification Report: \n", classification_report(y_test, y_pred))


# ####  Support Vector Machine

# In[42]:


svm = svm.SVC(kernel="linear")
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
print( 'Accuracy Score',accuracy_score(y_test, y_pred)*100)
print("\nClassification Report: \n", classification_report(y_test, y_pred))


# In[45]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

