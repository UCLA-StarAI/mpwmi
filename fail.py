
from pywmi import Density
from pympwmi import MPWMI3

from sys import argv

FAIL_PATH="fail.density"

density = Density.from_file(FAIL_PATH)
f, w = density.support, density.weight


if int(argv[1]) == 0:
    num_mpwmi = MPWMI3(f, w, msgtype='numeric')
    num_Z, _ = num_mpwmi.compute_volumes(cache=False)
    print(num_Z)
else:

    sym_mpwmi = MPWMI3(f, w, msgtype='symbolic')
    sym_Z, _ = sym_mpwmi.compute_volumes(cache=False)
    print(sym_Z)


#print(sym_Z, num_Z)

