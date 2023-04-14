# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:02:57 2023

@author: Jacob Rogers
"""
import pandas as pd
import numpy as np
import Phone_extract
import Http_extract
import Email_extract
import matplotlib.pyplot as plt

# pip install nltk
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# pip install sklearn
# Used to map words to numerical values 
from sklearn.feature_extraction.text import TfidfVectorizer

# Auto train/test data separation
from sklearn.model_selection import train_test_split
# Used to scale values
from sklearn.preprocessing import MaxAbsScaler

# pip install tensorflow
# Used for neural network
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os.path

class SpamClassifier:
    """ Classifier constructor. 
        Initializes: 
    """
    def __init__(self):
        self.df = None
        self.messages = None
        self.xTrain = None
        self.yTrain = None
        self.xTest = None
        self.yTest = None
        self.xTrainTensor = None
        self.yTrainTensor = None
        self.xTestTensor = None
        self.yTestTensor = None
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.scaler = MaxAbsScaler()
        self.model = Sequential()
        
        # Used in the word lemmatization. Download if needed, otherwise continue
        nltk.data.path.append("Datasets/wordnet")
        try:
          nltk.data.find("corpora/wordnet.zip")
        except LookupError:
             print("Downloading tools for language processing. This may take a few minutes.")
             nltk.download("wordnet", download_dir = "Datasets/wordnet/")     
        
        # Check if preformatted data file exists, if so, yay, we optimized! Otherwise
        # We have to dl og set, clean it, write it and carry on
        if not os.path.isfile("Datasets/classifyDataFrame.csv"):
           self.df = self._createData()
           # Transform strings through stemming and lemmatization
           self.vectorizer.fit(self.df.TEXT.tolist())
           self._cleanText()
           self.df.to_csv("Datasets/classifyDataFrame.csv", index = False)
        else:
            self.df = pd.read_csv("Datasets/classifyDataFrame.csv")
            # Transform strings through stemming and lemmatization
            self.vectorizer.fit(self.df.TEXT.tolist())
        
        # Define train/test split     
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.df.drop(columns = ["LABEL"]), self.df.LABEL, test_size=0.3, random_state=42)
        self.xTrainTensor, self.yTrainTensor, self.xTestTensor, self.yTestTensor = self._buildClassTensor()
        self.plotMessageDistribution()
        self._buildNNModel()
        
        
    def _createData(self):
        """ 
        DATA IMPORT SECTION.
        # LABEL: ham, spam, smish [convert to ham = 0, spam = 1, smishing = 2]
        # TEXT: The associated text message
        # URL: Whether a URL appears in the message or not [convert to False = 0 or True = 1]
        # EMAIL: Whether an email appears in the message or not [convert to False = 0 or True = 1]
        # PHONE: Whether a phone number appears in the message or not [convert to False = 0 or True = 1]

        We have two data files sources, one with a specific ham/spam format, the other with ham/spam/smishing.
        This section of code reads in both files into pandas data frames, formats the two data frames to have 
        comparable attributes and labels (calling extraction methods to idenitify URL, PHONE and EMAIL when needed),
        reformats the attributes to numerical values for supervised learning classification and plots distribution 
        of message types for visualization of data. Combines the two data frames to a single frame and defines 
        the training/testing split of overall data.
        """
        # Read in datasets
        # Courtesy of: https://data.mendeley.com/datasets/f45bkkt8pr
        df1 = pd.read_csv("Datasets/Dataset_5971.csv")

        df1.loc[df1['URL'] == 'No', 'URL'] = 0
        df1.loc[df1['URL'] == 'yes', 'URL'] = 1
        df1.loc[df1['EMAIL'] == 'No', 'EMAIL'] = 0
        df1.loc[df1['EMAIL'] == 'yes', 'EMAIL'] = 1
        df1.loc[df1['PHONE'] == 'No', 'PHONE'] = 0
        df1.loc[df1['PHONE'] == 'yes', 'PHONE'] = 1


        # Courtesy of: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download
        # spam.csv has a strange encoding, have to specify encoding to read in properly
        df2 = pd.read_csv("Datasets/spam.csv", encoding = "Windows-1252", usecols = ["v1", "v2"])
        df2.rename(columns={"v1": "LABEL", "v2": "TEXT"}, inplace = True)

        # Extract PHONE, URL, EMAIL from text in df2
        phone = []
        url = []
        email = []
        for i in range(len(df2)):
            textAtIdx = df2.TEXT[i]
            # All extract methods return 0 for False, 1 for True. Append each to lists
            phone.append(Phone_extract.phoneNumber_check(textAtIdx))
            url.append(Http_extract.http_check(textAtIdx))
            email.append(Email_extract.email_check(textAtIdx))

        # store attributes into df
        df2["PHONE"] = phone
        df2["URL"] = url
        df2["EMAIL"] = email

        # concatenate both dfs into a single df
        df = pd.concat([df1, df2])

        # Ensure ham, spam, smish labels are all labeled correctly
        df.loc[df["LABEL"] == "Ham", "LABEL"] = "ham"
        df.loc[df["LABEL"] == "Spam", "LABEL"] = "spam"
        df.loc[df["LABEL"] == "Smishing", "LABEL"] = "smish"
        df.loc[df["LABEL"] == "smishing", "LABEL"] = "smish"
        # Map labels to numerical values
        df.loc[df["LABEL"] == "ham", "LABEL"] = 0
        df.loc[df["LABEL"] == "spam", "LABEL"] = 1
        df.loc[df["LABEL"] == "smish", "LABEL"] = 2
        # Remove duplicates from df.
        df.drop_duplicates(inplace = True, ignore_index = True)
        # Write dataframe to csv file for faster access later
        return df

    def plotMessageDistribution(self):

        # Message type percentage over all dist.
        spamPercentage = sum(self.df.LABEL == 1)/len(self.df)
        hamPercentage = sum(self.df.LABEL == 0)/len(self.df)
        smishPercentage = sum(self.df.LABEL == 2)/len(self.df)

        # Pie chart to show distribution of messages
        colors = ["purple", "darkgreen", "red"]
        plt.pie([spamPercentage, hamPercentage, smishPercentage], labels = ["Spam", "Ham", "Smishing"], autopct='%1.1f%%', colors = colors)
        plt.title("Distribution of Message Classification")
        plt.show()


    """
    TEXT CLEANING SECTION.

    stemmer: Used for word stemming, this converts a word to it's base root (the stem), by removing
    commonly used prefixes/suffixes. This should result in a reduced language bin size and a more effective 
    mapping of words to real valued numbers.

    lemmatizer: Used for word lemmatization. This converts words to a base dictionary word (lemma) without 
    losing the core identity/meaning of the initial word used. Again, should help reduce language bin size, 
    and provide a more accurate mapping of words to real valued numbers.

    This section defines the methods used to perform word stemming, lemmatization and complete text cleaning on
    a given text argument passed to the method and for a complete data frame passed in as an argument. It also 
    downloads needed packages for implementing lemmatization (wordnet corpus, the dictionary used to map words to lemma)
    """

    # Performs word stemming on the text argument and returns the cleaned text
    def textStemmer(self, text):
        stemmer = PorterStemmer()
        stemmedText = [stemmer.stem(word) for word in text]    
        return "".join(stemmedText)

    # performs word lemmatization on the text argument and returns the cleaned text
    def textLemmatizer(self, text):
        lemmatizer = WordNetLemmatizer()
        lemmaText = [lemmatizer.lemmatize(word) for word in text]
        return "".join(lemmaText)

    # Performs text lemmatization and word stemming on all text attributes for the given
    # pandas data frame argument
    def _cleanText(self):
        # Clean all text attribute values and replace with new values
        for i in range(len(self.df)):
            self.df.TEXT[i] = self.textLemmatizer(self.df.TEXT[i])
            self.df.TEXT[i] = self.textStemmer(self.df.TEXT[i])
    
    def text2Vec(self, df):
        # TfidfVectorizer encodes words to a higher dimensional space, separating the numerical
        # plane mapping for given words
        transformMatrix = self.vectorizer.transform(df.TEXT.tolist())
        return transformMatrix
    
    def scaleData(self, data):
        return self.scaler.transform(data)
            
    def _buildClassTensor(self):
        # Build word mapped to value matrix
        xTrainMtx = self.text2Vec(self.xTrain)
        xTestMtx = self.text2Vec(self.xTest)
        
        # scale data
        xTrainMtx = self.scaler.fit_transform(xTrainMtx)
        xTestMtx = self.scaleData(xTestMtx)
        
        # drop og text as it's not used
        self.xTrain = self.xTrain.drop(columns = ["TEXT"])
        self.xTest = self.xTest.drop(columns = ["TEXT"])

        # convert columns to numpy array and reshape to (n, 1)
        urlTrain = np.array(self.xTrain['URL']).reshape(-1, 1)
        emailTrain = np.array(self.xTrain['EMAIL']).reshape(-1, 1)
        phoneTrain = np.array(self.xTrain['PHONE']).reshape(-1, 1)
        urlTest = np.array(self.xTest['URL']).reshape(-1, 1)
        emailTest = np.array(self.xTest['EMAIL']).reshape(-1, 1)
        phoneTest = np.array(self.xTest['PHONE']).reshape(-1, 1)

        # concatenate text matrix with other columns
        xTrainMtx = np.concatenate((xTrainMtx.toarray(), urlTrain, emailTrain, phoneTrain), axis=1)
        xTestMtx = np.concatenate((xTestMtx.toarray(), urlTest, emailTest, phoneTest), axis=1)
        
        # Convert matrice to tensors as input into NN
        xTrainTensor = tf.convert_to_tensor(xTrainMtx.astype("float32"))
        yTrainTensor = tf.convert_to_tensor(self.yTrain.astype("float32"))
        xTestTensor = tf.convert_to_tensor(xTrainMtx.astype("float32"))
        yTestTensor = tf.convert_to_tensor(self.yTrain.astype("float32"))
        return(xTrainTensor, yTrainTensor, xTestTensor, yTestTensor)


    """ 
    CLASSIFICATION SECTION.
    
    We're using a deep neural network as our classifier. The peformance outweighs efficiency of 
    other basic classifiers (Naive Bayes/SGD) and is tuned to be fairly efficient computationally.
    The initial layer accepts the input shape of the training tensor (this is the number of inputs 
    going into a black box that does some maths to compute an output).
    The activation of each layer is the maths of the black box, using relu: 
    a linear function that return max(x, 0). Each layer halves the number of inputs into the next layer
    until we reach the final layer, with an output of 3 (an output for each type of message: 0, 1, 2).
    Softmax calculates the probability distribution of the given NN outputs over the output class types.
    
    The number of epochs can be varied between 10-20, to produce reliable and efficient results.
    The batch size helps improve efficiency with a minor reduction in accuracy. Recommend to leave alone.

    """
    def _buildNNModel(self, epochs = 15, batch_size = 750):
        self.model.add(Dense(256, input_shape=(self.xTrainTensor.shape[1],), activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation = "relu"))
        # Final layer. Output 3 nodes with softmax eq to calc prob dist over all classes
        self.model.add(Dense(3, activation="softmax"))

        print("Model created.")
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

        print("Model compiled.")

        self.model.fit(self.xTrainTensor, self.yTrainTensor, epochs = epochs, batch_size = batch_size)
        print("Model Fitted.")

        test_loss, test_acc = self.model.evaluate(self.xTestTensor,  self.yTestTensor, verbose=2)
        print('\nTest accuracy:', test_acc)
        
        
sc = SpamClassifier()







