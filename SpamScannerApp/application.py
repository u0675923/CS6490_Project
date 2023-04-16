from kivy.app import App
from kivy.core.window import Window
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock

import time
import threading

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

            # update button image
            self.ids.status_image.source = self.on_status
        else:
            self.program_running = False
            
            Clock.unschedule(self.UpdateTimeLabel)
            self.elapsed_time = 0
            self.ids.box.ids.program_runtime.data = "00d 00m 00s"

            # update button image
            self.ids.status_image.source = self.off_status
    
    def UpdateTimeLabel(self, dt):        
        elapsed_time = time.time() - self.start
        days, r = divmod(elapsed_time, 86400)
        hours, r = divmod(r, 3600)
        min, sec = divmod(r, 60)
        self.ids.box.ids.program_runtime.data = f"{int(days):02d}d {int(min):02d}m {int(sec):02d}s"

class MyApp(App):
    def build(self):
        Builder.load_file('application.kv')
        Config.set('graphics', 'width', '360')
        Config.set('graphics', 'height', '640')

        Window.size = (360,640)
        Window.clearcolor = (52/255,57/255,68/255,1) # RGBA value
        
        sm = ScreenManager()

        # create screens for the application
        main_screen = MainScreen(name='main_screen')

        # add screens to the screen manager
        sm.add_widget(main_screen)

        return sm
