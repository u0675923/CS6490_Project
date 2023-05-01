import time
import os

from kivy.app import App
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from android.permissions import request_permissions, Permission
from jnius import autoclass, cast
from kivy.utils import platform


# Set up the spam scanner service
PythonActivity = autoclass('org.kivy.android.PythonActivity')
spam_scanner_service = autoclass("org.test.spamscanner.ServiceSpamscanner")


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

            # Start the service
            spam_scanner_service.start(PythonActivity.mActivity, "")

            self.ids.status_image.source = self.on_status
        else:
            self.program_running = False

            # Stop the service
            spam_scanner_service.stop(PythonActivity.mActivity)

            self.ids.status_image.source = self.off_status

    
    # Buttons for switching screens
    def AboutButtonPressed(self):        
        self.manager.current = 'about_screen'

class AboutScreen(Screen):
    back_button = 'Images/back_button.png'
    back_button_pressed = 'Images/pressed_back_button.png'

    def BackButtonPressed(self):
        self.ids.back_button.source = self.back_button_pressed

    def BackButtonReleased(self):
        self.ids.back_button.source = self.back_button
        self.manager.current = 'main_screen'


class MyApp(App):
    def build(self):
        # Attempt to request permissions
        permissions = [
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_SMS,
            Permission.FOREGROUND_SERVICE,
            Permission.POST_NOTIFICATIONS,
        ]

        # Request the permissions
        request_permissions(permissions)

        # load the kivy design files
        Builder.load_file('main_screen.kv')
        Builder.load_file('about.kv')

        Window.clearcolor = (52/255,57/255,68/255,1) # RGBA value
        
        sm = ScreenManager()

        # create screens for the application
        main_screen = MainScreen(name='main_screen')
        about_screen = AboutScreen(name='about_screen')

        # add screens to the screen manager
        sm.add_widget(main_screen)
        sm.add_widget(about_screen)

        return sm


if __name__ == '__main__':
    MyApp().run()
