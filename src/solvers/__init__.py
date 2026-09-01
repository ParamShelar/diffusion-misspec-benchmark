from . import dps, diffpir, pigdm
from .baselines import zero_filled, tv

SOLVERS = {"dps": dps.solve, "diffpir": diffpir.solve, "pigdm": pigdm.solve,
           "zero_filled": zero_filled, "tv": tv}
