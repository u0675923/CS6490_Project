from time import sleep

from kvdroid.tools.sms import get_all_sms

from jnius import autoclass

# PythonService = autoclass('org.kivy.android.PythonService')
# PythonService.mService.setAutoRestartService(True)

print(get_all_sms())

while True:
    print("Service still running...")
    sleep(5)
