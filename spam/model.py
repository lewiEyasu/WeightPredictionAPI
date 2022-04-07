import collections
import pandas as pd 
import numpy as np
import json 
import os
from django.conf import settings


class Spam_model:


    def __init__(self, message):

        self.message = message
        self.data = {}
        self.dictionary  = {}
        self.model_param = {}
    
    def load_data(self):
        
        with open(settings.MODELS ,'r') as f:
             self.data = json.load(f)  
        phi_y = float(self.data["phi_y"])
        phi_k_y1 = np.array(self.data["phi_k_y1"])
        phi_k_y0 =np.array(self.data["phi_k_y0"])
        self.model_param = {'phi_y': phi_y, 'phi_k_y0': phi_k_y0, 'phi_k_y1': phi_k_y1 }
        self.dictionary = self.data['dictionary']



    def encode_text(self,message):
        
        '''
        Args:
            messages: A List containing an SMS messages
            dictionary: A list of word in form in dictionary

        Returns:
        encode the messages refeing to the dictionary 
        
        '''
        self.load_data()
        n = len(self.dictionary)
        message = message.lower().split()
        words_count =collections.Counter(message)
        encode_text_matrix = np.zeros((1,n) , dtype=int)
        print("count ", encode_text_matrix.shape)
        for word, count in words_count.items():
            if word in self.dictionary:
                encode_text_matrix[0][self.dictionary[word]] += count
                    
                
        return encode_text_matrix  

    def predict_from_naive_bayes_model(self):

        matrix = self.encode_text(self.message)
        
        phi_y = self.model_param['phi_y']
        phi_k_y0 = self.model_param['phi_k_y0']
        print(matrix.shape)
        phi_k_y1 =  self.model_param['phi_k_y1']
        fit = matrix.dot (np.log(phi_k_y1) - np.log(phi_k_y0)) + np.log(phi_y / (1 - phi_y)) >= 0
        return 'Spam' if fit else 'Not Spam'


'''
print()

message = 'even my brother is not like to speak with me. they treat me like aids patent'  
test = Spam_model(message)
print(test.predict_from_naive_bayes_model())
'''


   