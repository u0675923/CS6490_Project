from time import sleep

from kvdroid.tools import get_resource
from kvdroid.tools.notification import create_notification
from kvdroid.tools.sms import get_all_sms

from jnius import autoclass


# Get the initial SMS amount
prev_sms_count, _ = get_all_sms()

def notification(sms):
    text = f'Sender: {sms["number"]}\nMessage: {sms["body"]}'

    create_notification(
        small_icon=get_resource('drawable').notification_template_icon_bg,
        channel_id='1',
        title="Message received",
        text=text,
        ids=1,
        channel_name='ch1',
    )


# Start scanning the SMS messages
while True:
    # Check if any new messages were received
    sms_count, sms_messages = get_all_sms()
    new_count = sms_count - prev_sms_count

    if new_count > 0:
        # Scan the new messages
        for i in range(new_count):
            if sms_messages[i]['type'] == 'inbox':
                notification(sms_messages[i])

        prev_sms_count = sms_count

    sleep(1)
