from time import sleep

from kvdroid.tools.sms import get_all_sms

from jnius import autoclass


# Get the initial SMS amount
prev_sms_count, _ = get_all_sms()


# Start scanning the SMS messages
while True:
    # Check if any new messages were received
    sms_count, sms_messages = get_all_sms()
    new_count = sms_count - prev_sms_count

    if new_count > 0:
        # Print out the new message(s)
        for i in range(new_count):
            if sms_messages[i]['type'] == 'inbox':
                print("Message received:")
                print(f'Date: {sms_messages[i]["date"]}')
                print(f'Sender: {sms_messages[i]["number"]}')
                print(f'Message: {sms_messages[i]["body"]}')

        prev_sms_count = sms_count

    sleep(1)
