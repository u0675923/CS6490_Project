import re
import glob
phonePattern = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')

def phoneNumber_check(line):
    if phonePattern.search(line) is not None:
        return 1
    else:
        return 0

