import time
import os

from kivy.app import App
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from android.permissions import request_permissions, Permission
from jnius import autoclass, cast
from kivy.utils import platform
import numpy as np
import re
import random
from assets import Phone_extract
from assets import Http_extract
from assets import Email_extract


# Used for classifying messages
ByteBuffer = autoclass('java.nio.ByteBuffer')
File = autoclass('java.io.File')
Interpreter = autoclass('org.tensorflow.lite.Interpreter')
TensorBuffer = autoclass('org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
Stemmer = autoclass('opennlp.tools.stemmer.PorterStemmer')

# Used to vectorize users sms
class smsTransformer:
    def __init__(self):
        self.stemmer = Stemmer()
        self.vocab = []
        self.idf_dict = {}
        # Load the vocabulary
        with open("assets/vocabFit.txt", 'r', encoding='utf-8') as f:
            self.vocab = [word.strip().replace('\n', '') for word in f.readlines()]

        # Load the IDF dictionary
        with open("assets/idfDictFit.txt", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    term, idf = line.rstrip().split()
                    self.idf_dict[term] = float(idf)
                except ValueError:
                # skip over any lines that don't have the expected format
                    continue

        
    def stem(self, word):
        self.stemmer.stem(word)
        wordStem = self.stemmer.toString()
        return str(wordStem)
    
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
    
    def transform(self, corpus):
        vectors = np.zeros((len(corpus), len(self.vocab)))
        for i, doc in enumerate(corpus):
            for j, term in enumerate(self.vocab):
                tf = doc.count(term) / len(doc)
            try:
                vectors[i, j] = tf * self.idf_dict[term]
            except KeyError:
                matching_keys = []
                for key in self.idf_dict:
                    if re.match(term[:2], key):
                        matching_keys.append(key)
                if matching_keys:
                    matching_keys = sorted(matching_keys, key=lambda x: len(x))
                    vectors[i, j] = tf * self.idf_dict[matching_keys[0]]
                elif len(term) > 1:
                    matching_keys = []
                    for key in self.idf_dict:
                        if re.match(term[:1], key):
                            matching_keys.append(key)
                    if matching_keys:
                        matching_keys = sorted(matching_keys, key=lambda x: len(x))
                        vectors[i, j] = tf * self.idf_dict[matching_keys[0]]
                    else:
                        random_key = random.choice(list(self.idf_dict.keys()))
                        vectors[i, j] = tf * self.idf_dict[random_key]
                else:
                    random_key = random.choice(list(self.idf_dict.keys()))
                    vectors[i, j] = tf * self.idf_dict[random_key]
        return vectors
    
    
    def maxabs_scale(self, X):
        max_abs = np.max(np.abs(X), axis=0)
        return X / max_abs
    
    def smsToVector(self, messages):
        phone, url, email = self._extractAttributesFromMessages(messages)
        vectors = self.transform(messages)
        vectors = self.maxabs_scale(vectors)
        # convert columns to numpy array and reshape to (n, 1)
        url = np.array(url).reshape(-1, 1)
        email = np.array(email).reshape(-1, 1)
        phone = np.array(phone).reshape(-1, 1)
        # concatenate text matrix with other columns
        return(np.concatenate((vectors, url, email, phone), axis=1))

    
class NN:
    def __init__(self):
        model = File(os.path.join(os.getcwd(), 'assets/quantizedModel.tflite'))
        self.interpreter = Interpreter(model)
        self.output_shape = self.interpreter.getOutputTensor(0).shape()
        self.output_type = self.interpreter.getOutputTensor(0).dataType()
        self.interpreter.allocateTensors()

    def _predictSms(self, vectors):
        # Prepare input data for tflite model
        input_data = np.array(vectors, dtype=np.float32)
        input_buffer = ByteBuffer.wrap(input_data.tobytes())
        # Prepare the output buffer
        output_buffer = TensorBuffer.createFixedSize(self.output_shape,
                                                     self.output_type)
        
        # Perform prediction
        self.interpreter.run(input_buffer, output_buffer.getBuffer().rewind())
        output_data = np.array(output_buffer.getFloatArray())
        if np.argmax(output_data) < 1:
            str = "ham"
        elif np.argmax(output_data) > 1.5:
            str = "phishing"
        else:
            str = "spam"
        return str
    
    def predict(self, vectors):
        str = ""

        for i in range(len(vectors) - 1):
            str += self._predictSms(vectors[i]) + ", "
        str += self._predictSms(vectors[i + 1])
        return str
    


# Used for getting the sms messages
PythonActivity = autoclass('org.kivy.android.PythonActivity')
Uri = autoclass('android.net.Uri')
Cursor = autoclass('android.database.Cursor')
smsVectorizer = smsTransformer()
classifier = NN()

class MainScreen(Screen):
    # images used in program:
    base_status =    'Images/base_button.png'
    pressed_status = 'Images/pressed_button.png'
    on_status =      'Images/on_button.png'
    off_status =     'Images/off_button.png'

    # indicator for running program
    program_running = False

    def StatusButtonPressed(self):
        self.ids.status_image.source = self.pressed_status

    def StatusButtonReleased(self):
        if not self.program_running:
            self.program_running = True

            # start program timer
            self.start = time.time()
            self.elapsed_time = 0
            Clock.schedule_interval(self.UpdateTimeLabel, 1)

            # run the classifier here
            x = np.array(np.random.random_sample((1, 35056)), np.float32)
            y = classifier.predict(x)
            print(f'Model output: {y}')

            # update button image
            self.ids.status_image.source = self.on_status
        else:
            self.program_running = False
            
            # stop program timer
            Clock.unschedule(self.UpdateTimeLabel)
            self.elapsed_time = 0
            self.ids.box.ids.program_runtime.data = "00m 00s"

            # update button image
            self.ids.status_image.source = self.off_status
    
    # Buttons for switching screens
    def AboutButtonPressed(self):        
        self.manager.current = 'about_screen'
    
    def SettingsButtonPressed(self):        
        self.manager.current = 'settings_screen'

    def ActivityButtonPressed(self):        
        self.manager.current = 'activity_screen'

    """
    Updates the timer label 
    """
    def UpdateTimeLabel(self, dt):        
        elapsed_time = time.time() - self.start

        # For now, just show minutes and seconds
        #days, r = divmod(elapsed_time, 86400)
        #hours, r = divmod(r, 3600)
        min, sec = divmod(elapsed_time, 60)
        self.ids.box.ids.program_runtime.data = f"{int(min):02d}m {int(sec):02d}s"

class AboutScreen(Screen):
    back_button = 'Images/back_button.png'
    back_button_pressed = 'Images/pressed_back_button.png'

    def BackButtonPressed(self):
        self.ids.back_button.source = self.back_button_pressed

    def BackButtonReleased(self):
        self.ids.back_button.source = self.back_button
        self.manager.current = 'main_screen'


class SettingsScreen(Screen):
    back_button = 'Images/back_button.png'
    back_button_pressed = 'Images/pressed_back_button.png'

    def BackButtonPressed(self):
        self.ids.back_button.source = self.back_button_pressed

    def BackButtonReleased(self):
        self.ids.back_button.source = self.back_button
        self.manager.current = 'main_screen'

class ActivityScreen(Screen):
    contentResolver = PythonActivity.mActivity.getContentResolver()

    back_button = 'Images/back_button.png'
    back_button_pressed = 'Images/pressed_back_button.png'

    def BackButtonPressed(self):
        self.ids.back_button.source = self.back_button_pressed

    def BackButtonReleased(self):
        self.ids.back_button.source = self.back_button
        self.manager.current = 'main_screen'

    # method that gets messages in phone
    def Do(self):
        output = ""
        
        # column names (you can add others like: phone number, date, contact name etc.)
        # **check docs for more information**
        projection = ['address', 'body'] # address prints the phone number

        # get the SMS content URI
        uri = Uri.parse('content://sms')

        # query the SMS table for all messages
        # docs on ContentResolver: https://developer.android.com/reference/android/content/ContentResolver
        cursor = self.contentResolver.query(uri, projection, None, None, None)

        # loop through the messages in the inbox
        while cursor.moveToNext():
            address = cursor.getString(cursor.getColumnIndex('address'))
            body = cursor.getString(cursor.getColumnIndex('body'))
            output += (f'{address}: {body}\n')

        # puts the ouput of the messages into the label in activity.kv
        tokens = ['testing', "You're Netflix account has been compromised! Please click the link and enter your password to save your account: https//www.test.co", "Hey, this is Kat, how you doing today??", "You could save a BUNDLE with testingNonsense car insurance today!"]
        # lemmTokens = smsTransformer.lemmatize(tokens)
        vectors = smsVectorizer.smsToVector(tokens)
        preds = classifier.predict(vectors)
        self.ids.test_label.text = str(preds)

class MyApp(App):
    def build(self):
        # Attempt to request permissions
        permissions = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_SMS]

        # Request the permissions
        request_permissions(permissions)

        # load the kivy design files
        Builder.load_file('main_screen.kv')
        Builder.load_file('activity.kv')
        Builder.load_file('settings.kv')
        Builder.load_file('about.kv')

        Window.clearcolor = (52/255,57/255,68/255,1) # RGBA value
        
        sm = ScreenManager()

        # create screens for the application
        main_screen = MainScreen(name='main_screen')
        about_screen = AboutScreen(name='about_screen')
        settings_screen = SettingsScreen(name='settings_screen')
        activity_screen = ActivityScreen(name='activity_screen')

        # add screens to the screen manager
        sm.add_widget(main_screen)
        sm.add_widget(about_screen)
        sm.add_widget(settings_screen)
        sm.add_widget(activity_screen)

        return sm

if __name__ == '__main__':
    MyApp().run()
