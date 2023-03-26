#!/usr/bin/env python
# coding: utf-8

# # Week 1 Assignment: Housing Prices
# 
# In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.
# 
# Imagine that house pricing is as easy as:
# 
# A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1 bedroom house cost 100k, a 2 bedroom house cost 150k etc.
# 
# How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.
# 
# Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

# In[1]:


import tensorflow as tf
import numpy as np
import keras


# In[2]:


def house_model():
    
    xs = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0])
    ys = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])

    model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    model.fit(xs,ys,epochs=1000)
    
    return model


# Now that you have a function that returns a compiled and trained model when invoked, use it to get the model to predict the price of houses: 

# In[3]:


# Get your trained model
model = house_model()


# Now that your model has finished training it is time to test it out! You can do so by running the next cell.

# In[4]:


new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)


# If everything went as expected you should see a prediction value very close to 4. **If not, try adjusting your code before submitting the assignment.** Notice that you can play around with the value of `new_x` to get different predictions. In general you should see that the network was able to learn the linear relationship between `x` and `y`, so if you use a value of 8.0 you should get a prediction close to 4.5 and so on.

# **Congratulations on finishing this week's assignment!**
# 
# You have successfully coded a neural network that learned the linear relationship between two variables. Nice job!
# 
# **Keep it up!**
