from datetime import datetime

def parse_tai_string(tstr):
    year   = int(tstr[:4])
    month  = int(tstr[4:6])
    day    = int(tstr[6:8])
    hour   = int(tstr[9:11])
    minute = int(tstr[11:13])
    return datetime(year, month, day, hour, minute)