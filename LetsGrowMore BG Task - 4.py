#!/usr/bin/env python
# coding: utf-8

# # LetsGrow More: Data Science Internship
# ## Task 4: (Beginner Level Task) Image to pencil sketch with Python
# ## Intern name : Koli Dhanashree Suresh

# ### Step 1 : Importing required modules or libraries

# In[1]:


#pip install opencv-python


# In[2]:


import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# ### Step 2 : Loading and Plotting Original Image

# In[7]:


image = cv2.imread(r'C:\Users\Admin\Downloads\wp10433310-sindhutai-sapkal-wallpapers.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")
plt.show()


# ### Step 3 : Converting Image to GrayScale

# In[8]:


image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8,8))
plt.imshow(image_gray,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.show()


# ### Step 4 : Inverting the grey scale image for better details

# In[9]:


image_invert = cv2.bitwise_not(image_gray)
plt.figure(figsize=(8,8))
plt.imshow(image_invert,cmap="gray")
plt.axis("off")
plt.title("Inverted Image")
plt.show()


# In[10]:


image_smoothing = cv2.GaussianBlur(image_invert, (21, 21),sigmaX=0, sigmaY=0)
plt.figure(figsize=(8,8))
plt.imshow(image_smoothing,cmap="gray")
plt.axis("off")
plt.title("Smoothened Image")
plt.show()


# ### Step 5 : Converting image to pencil sketches

# In[11]:


sketch = cv2.divide(image_gray,255-image_smoothing, scale=255)
plt.figure(figsize=(8,8))
plt.imshow(sketch,cmap="gray")
plt.axis("off")
plt.title("Pencilsketch Image")
plt.show()


# ### Step 6 : The Final Output

# In[12]:


plt.figure(figsize=(20,20))
plt.subplot(1,5,1)
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")
plt.subplot(1,5,2)
plt.imshow(image_gray,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.subplot(1,5,3)
plt.imshow(image_invert,cmap="gray")
plt.axis("off")
plt.title("Inverted Image")
plt.subplot(1,5,4)
plt.imshow(image_smoothing,cmap="gray")
plt.axis("off")
plt.title("Smoothened Image")
plt.subplot(1,5,5)
plt.imshow(sketch,cmap="gray")
plt.axis("off")
plt.title("Pencilsketch Image")
plt.show()


# ## Thank You...
