
import jnius
import kivy

if kivy.utils.platform == "android":
    Uri = jnius.autoclass('android.net.Uri')
    Cursor = jnius.autoclass('android.database.Cursor')
    PythonActivity = jnius.autoclass('org.kivy.android.PythonActivity')


class SMSReader:
    def __init__(self):
        """
        Constructor.
        """
        self.__context = None

        if kivy.utils.platform == "android":
            pass
