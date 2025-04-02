import pandas as pd
import numpy as np
import math

from instance_generator import generate_timetable
from data_structures import *

# Linha 4
tt_cp1_l4 = generate_timetable("CP1_L4", 290)
tt_cp2_l4 = generate_timetable("CP2_L4", 290, start_time="06:10")

# Linha 59
tt_cp1_l59 = generate_timetable("CP1_L59", 100)
tt_cp2_l59 = generate_timetable("CP2_L59", 100, start_time="06:00")

# Linha 60
tt_cp1_l60 = generate_timetable("CP1_L60", 120)
tt_cp2_l60 = generate_timetable("CP2_L60", 120, start_time="06:00")




dist = 0
time = 0

