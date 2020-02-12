
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pandas as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as py
get_ipython().magic('matplotlib inline')


# In[3]:


df=pd.read_csv("/home/ritesh/Desktop/dataset/glass.csv")
df.head()


# In[4]:


A=np.asmatrix(df)


# In[5]:


A_trans_A=np.matmul(A.T,A)
A_trans_A


# In[6]:


eig_value,eig_vect=np.linalg.eig(A_trans_A)


# In[7]:


eig_value


# In[8]:


index = eig_value.argsort()[::-1]
index


# In[9]:


Eig_value=eig_value[index]
Eig_value


# In[10]:


#singular Values

Sg_val=np.sqrt(Eig_value)
Sg_val


# In[11]:


#Diagnol Matrix

S=np.asmatrix(np.diag(Sg_val))
S


# In[12]:


#Inverse of diagnal matrix

S_inv=np.linalg.inv(S)
S_inv.shape


# In[13]:


V =eig_vect[:,index]
V


# In[14]:


#compute U=AVS_inv
U=A.dot(V).dot(S_inv)


# In[15]:


U


# In[16]:


U.shape


# In[17]:


error_list = []
for i in range(1,11):
    re_shapen = U[:,:i].dot(S[:i,:i]).dot(V[:,:i].T)
    error_list.append(np.linalg.norm(re_shapen-A))


# In[18]:


error_list


# In[24]:


error_list_2


# In[20]:


from sklearn.decomposition import TruncatedSVD


# In[21]:


error_list_2=[]
for i in range(1,10):
    svd = TruncatedSVD(n_components=i)
    pcs = svd.fit_transform(A)
    resh = svd.inverse_transform(pcs)
    error_list_2.append(np.linalg.norm(resh-A))


# In[22]:


x_1 = [i for i in range(1,10)]


# In[25]:


x=[i for i in range(1,11)]
plt.figure(figsize=(7,7))
plt.plot(x,error_list)
plt.plot(x_1,error_list_2)
plt.xlabel("SVD Component")
plt.ylabel("Error")
plt.plot()

