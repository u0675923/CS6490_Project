# Spam Scanner App

A project for CS6490: Network Security at the University of Utah.

## Authors

Adrian Flores  
Jake Rogers  
Seph Pace  


## Building the Project

This project is built using [Buildozer](https://buildozer.readthedocs.io/en/latest/),
which is a mobile application package compiler for Python. It can be installed
through [pip](https://buildozer.readthedocs.io/en/latest/installation.html) 
or built using Docker, with instructions on the [Buildozer source page](https://github.com/kivy/buildozer).

If installed with pip, the project can be built with the following command, which will place the project apk
file in the `bin/` directory.

```
$ buildozer android debug
```

If buildozer is installed with the Docker image, the project can be built with
the following command:

```
$ docker run --volume "$(pwd)":/home/user/hostcwd buildozer android debug
```

A utility script `build.sh` is there to build with docker more easily, in which
case you can just type:

```
$ ./build.sh android debug
```

## Installation

To install the project on an Android device, the generated apk file can simply
be moved to the device (through file transfer or uploading to and downloading
from the cloud) then opened on the device. It will ask for permission to
install and may warn that it is unsafe at which point you can press "install
anyway". Once installed the app will run and ask for certain permissions. Press
"accept" so that the app can function properly.


## Asset Sources

### Datasets

- [SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION](https://data.mendeley.com/datasets/f45bkkt8pr)
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)
- [Email Spam Dataset](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset)
- [Phishing Email Data by Type](https://www.kaggle.com/datasets/charlottehall/phishing-email-data-by-type)

### Images

- [On Off Button PNGs by Vecteezy](https://www.vecteezy.com/free-png/on-off-button)
- [https://www.iconsdb.com/white-icons/arrow-105-icon.html](https://www.iconsdb.com/white-icons/arrow-105-icon.html)
