from datetime import datetime
from sgp4.api import Satrec, SatrecArray
import numpy as np

line1 = "1 00900U 64063C   26053.30915840  .00000611  00000+0  61400-3 0  9995"
line2 = "2 00900  90.2154  69.1008 0023401 281.0538 102.8997 13.76475675 55628"
sat = Satrec.twoline2rv(line1, line2)
arr = SatrecArray([sat])

dt = datetime.utcnow()
diff = dt - datetime(1858, 11, 17)
jd_whole = diff.days + 2400000.5
jd_frac = diff.seconds / 86400.0 + diff.microseconds / 86400000000.0

e, r, v = arr.sgp4(np.array([jd_whole]), np.array([jd_frac]))
print("Array e:", e)
print("r:", r)
print("e dtype:", e.dtype)
