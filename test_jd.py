from datetime import datetime
def _datetime_to_jd(dt):
    year, month, day = dt.year, dt.month, dt.day
    hour, m, sec = dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
    if month <= 2:
        year -= 1; month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    fr = (hour + m / 60.0 + sec / 3600.0) / 24.0
    return jd, fr

dt = datetime.utcnow()
print(_datetime_to_jd(dt))
