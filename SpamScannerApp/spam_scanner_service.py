
import re
import random
from time import sleep

from jnius import autoclass

from kvdroid.tools import get_resource
from kvdroid.tools.notification import create_notification
from kvdroid.tools.sms import get_all_sms

import numpy as np

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
        if np.argmax(output_data) == 0:
            pred = "ham"
        elif np.argmax(output_data) == 2:
            pred = "phishing"
        else:
            pred = "spam"
        return pred

    def predict(self, vectors):
        pred = ""
        if len(vectors) == 1:
            pred += self._predictSms(vectors[0])
        else:
            for i in range(len(vectors) - 1):
                pred += self._predictSms(vectors[i]) + ", "
            pred += self._predictSms(vectors[i + 1])
        return pred


def notification(sms, classification):
    create_notification(
        small_icon=get_resource('drawable').notification_template_icon_bg,
        channel_id='1',
        title=f'{classification.upper()} received from {sms["number"]}',
        text=sms['body'],
        ids=1,
        channel_name='ch1',
    )


# Set up the classifier
smsVectorizer = smsTransformer()
classifier = NN()

# Get the initial SMS amount
prev_sms_count, _ = get_all_sms()

# Start scanning the SMS messages
while True:
    # Check if any new messages were received
    sms_count, sms_messages = get_all_sms()
    new_count = sms_count - prev_sms_count

    if new_count > 0:
        # Scan the new messages
        for i in range(new_count):
            if sms_messages[i]['type'] == 'inbox':
                vectors = smsVectorizer.smsToVector([sms_messages[i]['body']])
                preds = classifier.predict(vectors)

                if preds != 'ham':
                    notification(sms_messages[i], preds)

        prev_sms_count = sms_count

    sleep(1)
