# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:20:31 2023

@author: human
"""

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

# Auto train/test data separation
from sklearn.model_selection import train_test_split
# Used to scale values
from sklearn.preprocessing import MaxAbsScaler

# pip install tensorflow
# Used for neural network
import tensorflow as tf
# Deep NN model
from tensorflow.keras import Sequential
# Deep NN layer
from tensorflow.keras.layers import Dense
# Used to stop network if performance drops
from tensorflow.keras.callbacks import EarlyStopping
# Used for file/directory checks
import os.path
# Used to suppress a save trace warning 
import absl.logging
# Needed to save the sparse matrix for easier loading/fitting
from scipy.sparse import coo_matrix, save_npz, load_npz

absl.logging.set_verbosity(absl.logging.ERROR)


class SpamClassifier:
    """ 
    Classifier constructor. 
        Attempts to load pre-built neural network model, dataframe structures
        and initialize base components of the neural network and data cleansing system.
        If data can not be loaded, proceeds to build the components at each step,
        saving the base data in Datasets/classifyDataFrame.csv and Datasets/nnModel.
    """
    def __init__(self):
        self.scaler = MaxAbsScaler()
        self.idf_dict = {}
        self.vocab = []
    
        # Used in the word lemmatization. Download if needed, otherwise continue
        nltk.data.path.append("Datasets/wordnet")
        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            print("Downloading tools for language processing. This may take a few minutes.")
            nltk.download("wordnet", download_dir = "Datasets/wordnet/")
            
        self.interpreter = None
        if os.path.exists("Datasets/quantizedModel.tflite") and os.path.isfile("Datasets/classifyDataFrame.csv") and os.path.isfile("Datasets/scalerFitMtx.npz"):
            # Fit the needed objects for transformin/scaling user data
            self.df = pd.read_csv("Datasets/classifyDataFrame.csv")
            self.scaler.fit(load_npz("Datasets/scalerFitMtx.npz"))
            
            print("Model found. Loading model from memory...")
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path="Datasets/quantizedModel.tflite")
            print("Model loaded.")
                    
        else:
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
            self.xValidateTensor = None
            self.yValidateTensor = None
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
            if not os.path.isfile("Datasets/classifyDataFrame.csv") or not os.path.isfile("Datasets/scalerFitMtx.npz"):
                print("Initializing learning data. This may take awhile...")
                self.df = self._createData()
                self._cleanText(self.df)
                self.df.to_csv("Datasets/classifyDataFrame.csv", index = False)
            else:
                print("Loading in initialized learning data.")
                self.df = pd.read_csv("Datasets/classifyDataFrame.csv")
                # Transform strings through stemming and lemmatization
                
                
            # Define train/test split     
            self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.df.drop(columns = ["LABEL"]), self.df.LABEL, test_size=0.3, random_state=42)
            # Build tensors from mtx and train/test split
            self.xTrainTensor, self.yTrainTensor, self.xTestTensor, self.yTestTensor = self._buildClassTensor()
        
            # Define train/validation tesnor split. Must convert tensor to np array
            self.xTrainTensor, self.xValidateTensor, self.yTrainTensor, self.yValidateTensor = train_test_split(self.xTrainTensor.numpy(), self.yTrainTensor.numpy(), test_size=0.1, random_state=42)
            # then each numpy back to tensor. *eyeroll*
            self.xTrainTensor = tf.constant(self.xTrainTensor)
            self.xValidateTensor = tf.constant(self.xValidateTensor)
            self.yTrainTensor = tf.constant(self.yTrainTensor)
            self.yValidateTensor = tf.constant(self.yValidateTensor)
            # Message dist for eye candy
            self.plotMessageDistribution()
            # If classifier has already been built, load from memory
            if os.path.exists("Datasets/nnModel.h5"):
                print("Previous classifier found. Loading from memory.")
                self.model = tf.keras.models.load_model("Datasets/nnModel.h5")
                self._testNNModel()

                if os.path.exists("Datasets/quantizedModel.tflite"):
                    with open("Datasets/quantizedModel.tflite", "rb") as file:
                        quant = file.read()
                        self._testTFLiteModel()
                else:
                    self._quantizeModel()
                    # Build model from memory
            else:
                print("Building classifier model. Please wait.")
                self._buildNNModel()
                self._fitNNModel()
                self._testNNModel()
                self._quantizeModel()
                
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
            df.loc[i, "TEXT"] = self.textStemmer(df.TEXT[i])
            #df.loc[i, "TEXT"] = self.textLemmatizer(df.TEXT[i])
            
            
    # Transforms strings in dataframe to n-dimensional numerical vector representation
    def text2Vec(self, df):
        corpus = df.TEXT.tolist()
        if os.path.exists("Datasets/vocabFit.txt") and len(self.idf_dict) != 0:
            return self.transform(corpus)
        else:
            self.fit(corpus)
            return self.transform(corpus)

    
    def fit(self, corpus):
        n_docs = len(corpus)
        self.vocab = list(set([w for doc in corpus for w in doc]))
        for term in self.vocab:
            n_docs_with_term = sum([1 for doc in corpus if term in doc])
            self.idf_dict[term] = np.log(n_docs / (1 + n_docs_with_term))
            # Write the model to a file
            # Write the vocab to a file
        with open("Datasets/vocabFit.txt", 'w', encoding='utf-8') as f:
            for word in self.vocab:
                f.write(word + '\n')

        # Write the idf_dict to a file
        with open("Datasets/idfDictFit.txt", 'w', encoding='utf-8') as f:
            for term, idf in self.idf_dict.items():
                f.write(term + ' ' + str(idf) + '\n')  

    def transform(self, corpus):
        vectors = np.zeros((len(corpus), len(self.vocab)))
        for i, doc in enumerate(corpus):
            for j, term in enumerate(self.vocab):
                tf = doc.count(term) / len(doc)
                vectors[i, j] = tf * self.idf_dict[term]
        return vectors
    
    
    def maxabs_scale(self, X):
        max_abs = np.max(np.abs(X), axis=0)
        return X / max_abs
    
    # Scales given data through maxAbs operation
    def scaleData(self, data):
        return self.scaler.transform(data)
    
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
        return(np.concatenate((mtx, url, email, phone), axis=1))
        
    
    """
            Computes text to numerical vector transformation matrix for base data,
            scales transformed text matrix values, builds learning matrix through 
            concatenation of text2vec matrix and other base attributes and converts
            all train/test datasets to tensors (an n-dimensional set of matrices)
    """
    def _buildClassTensor(self):
        # Build word mapped to value matrix
        xTrainMtx = self.text2Vec(self.xTrain)
        xTestMtx = self.text2Vec(self.xTest)

        
        # scale data
        xTrainMtx = self.maxabs_scale(xTrainMtx)
       # xTrainMtx = self.scaler.fit_transform(xTrainMtx)
        xTestMtx = self.maxabs_scale(xTestMtx)
        # xTestMtx = self.scaleData(xTestMtx)
        
        # drop og text as it's not used
        self.xTrain = self.xTrain.drop(columns = ["TEXT"])
        self.xTest = self.xTest.drop(columns = ["TEXT"])

        # Concatenate tfid matrix with other attributes, forming one large data matrix
        xTrainMtx = self._concatMtxWithDf(xTrainMtx, self.xTrain)
        xTestMtx = self._concatMtxWithDf(xTestMtx, self.xTest)
        
        # Convert matrice to tensors as input into NN
        xTrainTensor = tf.convert_to_tensor(xTrainMtx.astype("float32"))
        yTrainTensor = tf.convert_to_tensor(self.yTrain.astype("float32"))
        xTestTensor = tf.convert_to_tensor(xTrainMtx.astype("float32"))
        yTestTensor = tf.convert_to_tensor(self.yTrain.astype("float32"))
        
        # Save xTrainTensor to fit v
        
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

    """
    def _buildNNModel(self):
        # Initial input layer
        self.model.add(Dense(256, input_shape=(self.xTrainTensor.shape[1],), activation="relu"))
        # Hidden layer 1 with 128 output neurons
        self.model.add(Dense(128, activation="relu"))
        # Hidden layer 2 with 64 output neurons
        self.model.add(Dense(64, activation = "relu"))
        # Final layer. Output 3 nodes with softmax eq to calc prob dist over all classes
        self.model.add(Dense(3, activation="softmax"))
        # Compile built model using adam optimizer and sparsecatcrossentropy for probability optimization
        self.model.compile(optimizer = "adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
    """
        Fits the network model to the tensors calculated earlier.
        The number of epochs can be varied, but default is at 100 with an early stop to produce reliable and efficient results.
        The batch size helps improve efficiency with a minor reduction in accuracy. Recommend to leave alone.
    """ 
    def _fitNNModel(self, epochs = 500, batch_size = 750):
        # Define a break point if loss on validation set drops and doesn't improve after 10 epochs
        earlyStop = EarlyStopping(monitor = "val_loss", patience = 10)
        # Fit model
        self.model.fit(self.xTrainTensor, self.yTrainTensor, epochs = epochs, batch_size = batch_size, validation_data=(self.xValidateTensor, self.yValidateTensor), callbacks=[earlyStop])
        # Save the Keras model to a directory
        self.model.save("Datasets/nnModel.h5")
        
        
        
    # Runs test on model based on fitted data and test data split
    def _testNNModel(self):
        test_loss, test_acc = self.model.evaluate(self.xTestTensor,  self.yTestTensor, verbose=2)
        print("\nTest accuracy:", test_acc)


    # Quantizes the model to save memory
    def _quantizeModel(self):
        # Quantize the model
        print("\nQuantizing the model...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_model_tflite = converter.convert()
        tf.io.write_file("Datasets/quantizedModel.tflite", quant_model_tflite)
        self.interpreter = tf.lite.Interpreter(model_path="Datasets/quantizedModel.tflite")
        self._testTFLiteModel()


    # Runs test on the given quantized model
    def _testTFLiteModel(self):
        self.interpreter.allocate_tensors()
        in_idx = self.interpreter.get_input_details()[0]["index"]
        out_idx = self.interpreter.get_output_details()[0]["index"]
        print("input shape: " , in_idx)
        print("output shape: " , in_idx)

        predicted_y = []
        for x in self.xTestTensor:
            x = np.expand_dims(x.numpy(), axis=0)
            self.interpreter.set_tensor(in_idx, x)
            self.interpreter.invoke()
            y = self.interpreter.tensor(out_idx)
            y = np.argmax(y()[0])
            predicted_y.append(y)
        quant_acc = (np.array(predicted_y) == self.yTestTensor.numpy()).mean()
        print("\nQuantized test accuracy: ", quant_acc)
        

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
    
    
    # Helper function that performs the cleaning process and formation of tensor for 
    # a given list of user's messages. Returns the user's input tensor.
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
        #userMtx = self.scaleData(userMtx)
        userMtx = self.maxabs_scale(userMtx)
        # form complete user mtx by concatenating other attributes with text2Vec
        userMtx = self._concatMtxWithDf(userMtx, userDf)
        # Convert matrice to tensors as input into NN
        userTensor = tf.convert_to_tensor(userMtx.astype("float32"))
        return userTensor
        
    # Converts user messages to acceptable tensors, and runs through NN to calc prob dist
    # of message type. Returns the highest prob message classification for each message as a list.
    def predictMessages(self, messageList):
        yPreds = []
        # Clean user's message and form tensor arg for NN
        userTensor = self._cleanUserMessage(messageList)
        # Return predictions of messages
        # return(np.argmax(self.model.predict(userTensor), axis=1))
               
        # Get input and output tensors.
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        print("User tensor shape: ", userTensor.shape)
        print("Expected input shape: ", input_details)
        for i in range(userTensor.shape[0]):
            # Set the input tensor
            self.interpreter.set_tensor(input_details[0]['index'],tf.expand_dims(userTensor[i], axis=0))
            # Run the inference
            self.interpreter.invoke()
            # Get the output tensor
            yPreds.append(np.argmax(self.interpreter.get_tensor(output_details[0]['index'])))
            
            
        return yPreds
        
        
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
        messageTensor = self._cleanUserMessage(messageList)
        # Create label tensor for training
        labelTensor = tf.convert_to_tensor(labelsList.astype("float32"))
        
        
        # Load the pre-trained model (if user has own model)
        if os.path.isfile("Datasets/nnModel.h5"):
            model = tf.keras.models.load_model('Datasets/nnModel.h5')
        else:
            print("No model to train!")
            
        # Freeze the layers of the pre-trained model to prevent them from being trained
        for layer in model.layers:
            layer.trainable = False
                        
        # Add two new hidden layer with ReLU activation
        hidden_layer = Dense(32, activation='relu')(model.layers[-2].output)
        hidden_layer2 = Dense(8, activation='relu')(hidden_layer)
        
        # Add a new output layer for the 3 classes to train
        new_output = Dense(3, activation='softmax')(hidden_layer2)

        # Create the new model with the new output layer
        new_model = tf.keras.Model(inputs=model.input, outputs=new_output)
        # Compile the model with an appropriate loss function and optimizer
        new_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

        # Train the model on the new data points
        new_model.fit(messageTensor, labelTensor, epochs = 20, batch_size=1)
        #new_model.save("Datasets/nnUserModel.h5")

        
        # Save user quantized model
        converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_model_tflite = converter.convert()
        tf.io.write_file("Datasets/quantizedModel.tflite", quant_model_tflite)
        self.interpreter = tf.lite.Interpreter(model_path="Datasets/quantizedModel.tflite")
        
    
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

sc = SpamClassifier()
# # We are hoping to acheive [1 1 1 1 1 2]
print("Predicted classification of messages: ", sc.predictMessages(messages))

# Form dictionary with correct label mapping
appendDict = {messages[0] : 1, messages[1] : 1, messages[2] : 1, messages[3] : 1, messages[4]: 1, messages[5] : 2}
# Test append to trainer
#sc.appendToTrainer(appendDict)
# predict on messages once more
sc._testTFLiteModel()





