import time

from kivy.app import App
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock

from android.permissions import request_permissions, Permission
from jnius import autoclass
from kivy.utils import platform
import numpy as np


# Used to vectorize users sms
class TfidfVectorizer:
    def __init__(self):
        self.idf_dict = {}
        
    def fit_transform(self, corpus):
        self.documents = corpus
        self.word_count = sum(len(document) for document in corpus)
        term_count = {}
        
        for document in corpus:
            for term in document:
                term_count[term] = term_count.get(term, 0) + 1
                
        for term, count in term_count.items():
            self.idf_dict[term] = np.log(self.word_count / count)
            
        return self.transform(corpus)
    
    def transform(self, corpus):
        vectors = []
        
        for document in corpus:
            vector = np.zeros(len(self.idf_dict))
            
            for i, term in enumerate(self.idf_dict.keys()):
                count = document.count(term)
                tf = count / len(document)
                idf = self.idf_dict[term]
                vector[i] = tf * idf
                
            vectors.append(vector)
        return vectors


# Used for classifying messages
ByteBuffer = autoclass('java.nio.ByteBuffer')
File = autoclass('java.io.File')
Interpreter = autoclass('org.tensorflow.lite.Interpreter')
TensorBuffer = autoclass('org.tensorflow.lite.support.tensorbuffer.TensorBuffer')

class NN:
    def __init__(self):
        model = File(os.path.join(os.getcwd(), 'quantizedModel.tflite'))
        self.interpreter = Interpreter(model)
        self.output_shape = self.interpreter.getOutputTensor(0).shape()
        self.output_type = self.interpreter.getOutputTensor(0).dataType(    )
        self.interpreter.allocateTensors()

    def predict(self, vectors):
        # Prepare input data for tflite model
        input_data = np.array(vectors, dtype=np.float32)
        input_buffer = ByteBuffer.wrap(input_data.tobytes())

        # Prepare the output buffer
        output_buffer = TensorBuffer.createFixedSize(self.output_shape,
                                                     self.output_type)
        
        # Perform prediction
        self.interpreter.run(input_buffer, output_buffer.getBuffer().rewind())
        output_data = np.array(output_buffer.getFloatArray())
        
        return output_data.tolist()
    


# Used for getting the sms messages
PythonActivity = autoclass('org.kivy.android.PythonActivity')
Uri = autoclass('android.net.Uri')
Cursor = autoclass('android.database.Cursor')
smsVectorizer = TfidfVectorizer()
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
        self.ids.test_label.text = output

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
