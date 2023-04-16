import tkinter as tk

from kivy.app import App
from kivy.core.window import Window
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager

class MainScreen(Screen):
    # images used in program:
    base_status =    'Images/base_button.png'
    pressed_status = 'Images/pressed_button.png'
    on_status =      'Images/on_button.png'
    off_status =     'Images/off_button.png'

    # indicator for running program
    program_running = False

    Config.set('graphics', 'width', '360')
    Config.set('graphics', 'height', '640')

    Window.size = (360,640)
    Window.clearcolor = (52/255,57/255,68/255,1) # RGBA value

    def StatusButtonPressed(self):
        self.ids.status_image.source = self.pressed_status

    def StatusButtonReleased(self):
        if not self.program_running:
            self.program_running = True
            self.ids.status_image.source = self.on_status
        else:
            self.program_running = False
            self.ids.status_image.source = self.off_status

class MyApp(App):
    def build(self):
        Builder.load_file('application.kv')
        
        sm = ScreenManager()

        # create screens for the application
        main_screen = MainScreen(name='main_screen')

        # add screens to the screen manager
        sm.add_widget(main_screen)

        return sm
