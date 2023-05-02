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
# Multinomial naive bayes model
from sklearn.naive_bayes import MultinomialNB


# Used for file/directory checks
import os.path
# Used to suppress a save trace warning 
import absl.logging
import pickle


absl.logging.set_verbosity(absl.logging.ERROR)


class BayesClassifier:
    """ 
    Classifier constructor. 
        Attempts to load pre-built neural network model, dataframe structures
        and initialize base components of the neural network and data cleansing system.
        If data can not be loaded, proceeds to build the components at each step,
        saving the base data in Datasets/classifyDataFrame.csv and Datasets/nnModel.
    """
    def __init__(self):
        
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.scaler = MaxAbsScaler()
        self.df = None
        self.xTrain = None
        self.yTrain = None
        self.xTest = None
        self.yTest = None
    
        # Used in the word lemmatization. Download if needed, otherwise continue
        nltk.data.path.append("Datasets/wordnet")
        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            print("Downloading tools for language processing. This may take a few minutes.")
            nltk.download("wordnet", download_dir = "Datasets/wordnet/")
            
        # Load existing model in for a user.
        if os.path.exists("Datasets/bayesClf.pkl") and os.path.isfile("Datasets/classifyDataFrame.csv"):
            # Fit the needed objects for transformin/scaling user data
            self.df = pd.read_csv("Datasets/classifyDataFrame.csv")
            self.vectorizer.fit(self.df.TEXT.tolist())
            print("Model found. Loading model from memory...")
            # Load the naive bayes model
            with open("Datasets/bayesClf.pkl", "rb") as f:
                self.clf = pickle.load(f)
            print("Model loaded.")
                    
        else:
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
                print("Initializing learning data. This may take awhile...")
                self.df = self._createData()
                # Transform strings through stemming and lemmatization
                self.vectorizer.fit(self.df.TEXT.tolist())
                self._cleanText(self.df)
                self.df.to_csv("Datasets/classifyDataFrame.csv", index = False)
            else:
                print("Loading in initialized learning data.")
                self.df = pd.read_csv("Datasets/classifyDataFrame.csv")
                # Transform strings through stemming and lemmatization
                self.vectorizer.fit(self.df.TEXT.tolist())
                 
            # If classifier has already been built, load from memory
            if os.path.exists("Datasets/bayesClf.pkl"):
                print("Previous classifier found. Loading from memory.")
                # load the saved classifier
                with open("Datasets/bayesClf.pkl", "rb") as f:
                    self.clf = pickle.load(f)
                self._testBayesModel()
            else:
                print("Building classifier model. Please wait.")
                self._buildBayesModel()
                
    """
            Accepts a list of message arguments. Scans each message for a URL, EMAIL,
            or Phone inside the message. For each message, appens a bitwise boolean argument, 
            False = 0, True = 1 for the existence of attribute for message i. Returns the 
            list of boolean values for each attribute as a list, return order:
                PHONE, URL, EMAIL.
    """            
    def _extractAttributesFromMessages(self, messages):
        # Extract PHONE, URL, EMAIL from text messages[i]
        phone = []
        url = []
        email = []
        for i in range(len(messages)):
            textAtIdx = messages[i]
            # All extract methods return 0 for False, 1 for True. Append each to lists
            phone.append(Phone_extract.phoneNumber_check(textAtIdx))
            url.append(Http_extract.http_check(textAtIdx))
            email.append(Email_extract.email_check(textAtIdx))
        return(phone, url, email)
        
        
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
        
        # Courtesy of: https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset
        df3 = pd.read_csv("Datasets/completeSpamAssassin.csv", usecols = ["Body", "Label"], dtype={'Body': str})
        df3.rename(columns={"Body": "TEXT", "Label": "LABEL"}, inplace = True)
        df3 = df3.dropna(subset=["TEXT"])
        
        # Courtesy of: https://www.kaggle.com/datasets/charlottehall/phishing-email-data-by-type
        df4 = pd.read_csv("Datasets/phishing_data_by_type.csv", usecols = ["Text", "Type"])
        df4.rename(columns = {"Text" : "TEXT", "Type" : "LABEL"}, inplace = True)
        df4.dropna(subset = ["TEXT"])
        
        # Extract PHONE, URL, EMAIL from text in df2
        phone, url, email = self._extractAttributesFromMessages(df2.TEXT.tolist())
        
        # store attributes into df
        df2["PHONE"] = phone
        df2["URL"] = url
        df2["EMAIL"] = email
                
        # Extract all spam messages from df3
        df3 = df3[df3["LABEL"] == 1]
        # Extract PHONE, URL, EMAIL from text in df3
        phone, url, email = self._extractAttributesFromMessages(df3.TEXT.tolist())
        
        # store attributes into df
        df3["PHONE"] = phone
        df3["URL"] = url
        df3["EMAIL"] = email
        
        # Extract all phishing emails from df4
        df4 = df4[df4["LABEL"] == "Phishing"]
        # Extract PHONE, URL, EMAIL from text in df4
        phone, url, email = self._extractAttributesFromMessages(df4.TEXT.tolist())
        
        # store attributes into df
        df4["PHONE"] = phone
        df4["URL"] = url
        df4["EMAIL"] = email
        
        # concatenate all dfs into a single df
        df = pd.concat([df1, df2, df3, df4])

        # Ensure ham, spam, smish labels are all labeled correctly
        df.loc[df["LABEL"] == "Ham", "LABEL"] = "ham"
        df.loc[df["LABEL"] == "Spam", "LABEL"] = "spam"
        df.loc[df["LABEL"] == "Smishing", "LABEL"] = "smish"
        df.loc[df["LABEL"] == "Phishing", "LABEL"] = "smish"
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
    def _cleanText(self, df):
        # Clean all text attribute values and replace with new values
        for i in range(len(df)):
            df.loc[i, "TEXT"] = self.textLemmatizer(df.TEXT[i])
            df.loc[i, "TEXT"] = self.textStemmer(df.TEXT[i])
            
            
    # Transforms strings in dataframe to n-dimensional numerical vector representation
    def text2Vec(self, df):
        # TfidfVectorizer encodes words to a higher dimensional space, separating the numerical
        # plane mapping for given words
        transformMatrix = self.vectorizer.transform(df.TEXT.tolist())
        return transformMatrix
    
    
    """
        Concatenates the three other attributes of df (URL, PHONE, EMAIL) with
        the sparse matrix that was generated from transforming the text in df to
        an n-dimensional vector count of mapped words, forming one complete sparse matrix.
    """
    def _concatMtxWithDf(self, mtx, df):
        # convert columns to numpy array and reshape to (n, 1)
        url = np.array(df["URL"]).reshape(-1, 1)
        email = np.array(df["EMAIL"]).reshape(-1, 1)
        phone = np.array(df["PHONE"]).reshape(-1, 1)
        # concatenate text matrix with other columns
        return(np.concatenate((mtx.toarray(), url, email, phone), axis=1))
        
    
    """
            Computes text to numerical vector transformation matrix for base data,
            scales transformed text matrix values, builds learning matrix through 
            concatenation of text2vec matrix and other base attributes and converts
            all train/test datasets to tensors (an n-dimensional set of matrices)
    """
    def _buildClassMatrices(self):
        # Build word mapped to value matrix
        xTrainMtx = self.text2Vec(self.xTrain)
        xTestMtx = self.text2Vec(self.xTest)
        
        # scale data
        xTrainMtx = self.scaler.fit_transform(xTrainMtx)
        xTestMtx = self.scaler.fit_transform(xTestMtx)
        
        # drop og text as it's not used
        self.xTrain = self.xTrain.drop(columns = ["TEXT"])
        self.xTest = self.xTest.drop(columns = ["TEXT"])

        # Concatenate tfid matrix with other attributes, forming one large data matrix
        xTrainMtx = self._concatMtxWithDf(xTrainMtx, self.xTrain)
        xTestMtx = self._concatMtxWithDf(xTestMtx, self.xTest)
        
        return(xTrainMtx, xTestMtx)


    """ 
    CLASSIFICATION SECTION.
    
        We're using a Naive multinomial Baye's net classifier.
        A Baye's classifier is one in which uses Baye's theorem: p(y|x) = p(y) * Π p(x_i|y)^{x_i}
        to compute the probability distribution of an event given some known probability dist.
        We are using the multinomial variation, which assumes a multinomial (discreet) distribution
        This is particularly effective as we are generating a multinomial distribution
        by the way we transform the strings to word vectors mapped in n dimensions.        

    """
    def _buildBayesModel(self):
        # Ensure train/test data is loaded in
        self._loadTrainTestSet()
        # Initialize and fit model
        self.clf = MultinomialNB(class_prior = [0.638, 0.283, 0.079])
        
        
        class_freq = np.array([0.638, 0.283, 0.079])
        class_weight = 1 / class_freq
        class_weight_norm = class_weight / np.sum(class_weight)
        
        weights = np.ones(self.xTrain.shape[0])
        indices = np.where(self.yTrain == 0)[0]
        weights[indices] = class_weight_norm[0]
        indices = np.where(self.yTrain == 1)[0]
        weights[indices] = class_weight_norm[1]
        indices = np.where(self.yTrain == 2)[0]
        weights[indices] = class_weight_norm[2]
        
        self.clf.fit(self.xTrain, self.yTrain, sample_weight = weights)
        # save the classifier to a file
        with open("Datasets/bayesClf.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        # Run test on built model
        self._testBayesModel()

    # Runs test on model based on fitted data and test data split
    def _testBayesModel(self):
        # Ensure train/test data is loaded in
        self._loadTrainTestSet()
        
        print("Score on training dataset: ", self.clf.score(self.xTrain, self.yTrain))
        print("Score on testing dataset: ", self.clf.score(self.xTest, self.yTest))

    def _loadTrainTestSet(self):
        if(self.xTrain is None):
            # Define train/test split     
            self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.df.drop(columns = ["LABEL"]), self.df.LABEL, test_size=0.3, random_state=42)
            # Build tensors from mtx and train/test split
            self.xTrain, self.xTest = self._buildClassMatrices()
            # Message dist for eye candy
            self.plotMessageDistribution()
        

    """
    USER FUNCTIONALITY SECTION.
    
        Provide the user the ability to predict the classification of messages 
        (Passed as a list argument) using the built in model [predictMessages].
        
        Provide the user the ability to help train the model based on selected messages.
        NOTE: We do not need to store the user's messages, we simply use the text + label
        provided by the user (as a mapped dictionary, such that {"Message Text" : "Label"})
        to further train the already existent model and then discard the message.
        This could lead to catastrophic forgetting and perhaps saving the messages to the dataframe
        is acceptable, but some user's probably wont like that.
    """
    
    
    # Helper function that performs the cleaning process and formation of matrix for 
    # a given list of user's messages. Returns the user's input matrix.
    # WARNING: DO NOT USE WITH USER'S LABEL. ONLY ATTRIBUTES.
    def _cleanUserMessage(self, messageList):
        # Extract phone, url, email attributes for each message
        phone, url, email = self._extractAttributesFromMessages(messageList)
        # build dictionary as base for a pandas dataframe
        userDict = {"TEXT" : messageList, "URL" : url, "EMAIL": email, "PHONE": phone}
        # Init user df from dict
        userDf = pd.DataFrame(data = userDict)
        
        # Clean each message (word stemm and lemmatization) 
        self._cleanText(userDf)
        
        # Transform user text to matrix
        userMtx = self.text2Vec(userDf)
        # Scale data
        userMtx = self.scaler.fit_transform(userMtx)
        # form complete user mtx by concatenating other attributes with text2Vec
        userMtx = self._concatMtxWithDf(userMtx, userDf)
        
        return userMtx
        
    # Converts user messages to acceptable tensors, and runs through NN to calc prob dist
    # of message type. Returns the highest prob message classification for each message as a list.
    def predictMessages(self, messageList):
        # Clean user's message and form tensor arg for NN
        userMtx = self._cleanUserMessage(messageList)
        return(list(self.clf.predict(userMtx)))
        
        
    """
        Takes in a dictionary mapping of a message to a classification type:
        0 = ham, 1 = spam, 2 = phishing and continues training model using these messages.
        Dictionary structure should be as follows: {"Message1" : 0, "Message2" : 2}
    """
    def appendToTrainer(self, messageDict):
        # extract messages from dict
        messageList = list(messageDict.keys())
        # extract labels from dict
        labelsList = np.asarray(list(messageDict.values()))
        # Create message tensor for training
        messageMtx = self._cleanUserMessage(messageList)
        # Perform fit (continues training from previous weights)
        self.clf.partial_fit(messageMtx, labelsList)
        
        # save the classifier to a file
        with open("Datasets/bayesClf.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        
    
""" 
     TESTING SECTION
"""
    
# # These are real text messages received by Jake and his wife. Used for testing purposes
messages = ["Hello, my Maria , nice to meet you. I am looking for a gentle, honest, and ambitious boyfriend. I like men who are honest and gentlemen. A good relationship comes from friendship. If you are interested, you can add me on Whatsapp: +16723002063 to share daily life together, better get to know each other better", 
            "Please call me",
            "You said you would answer me when I text? 3862154333 Kathy",
            "Hi! Its Austin. Theres private health plans available for spring enrollment! Low deductibles & flexible nationwide coverage. Can I send updated rates? or stop",
            "You still have time left to enroll! There's private healthcare plans available that can start ASAP. Dental & vision too. Can I send you a quick quote? or stop"]        

# # This is a phishing-esque style message generated by chatgpt, lmao he can be tricked to create phishing message, despite his 'ethical' programming.
messages.append("Dear customer, we are pleased to inform you that you have been selected for a special offer! Click the link below to claim your discount on our latest product line. Don't miss out on this amazing opportunity! https://fakeGeneratedlink.co")
# Form dictionary with correct label mapping
appendDict = {messages[0] : 1, messages[1] : 1, messages[2] : 1, messages[3] : 1, messages[4]: 1, messages[5] : 2}
bc = BayesClassifier()
bc._testBayesModel()
print("Predictions: ", bc.predictMessages(messages))
# print("Training classifier on messages.")
# bc.appendToTrainer(appendDict)
# print("Predictions after retraining model: ", bc.predictMessages(messages))
# bc._testBayesModel()


