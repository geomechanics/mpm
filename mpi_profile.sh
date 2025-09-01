#!/bin/bash
mpiexec -n 8 ./mpm -p 4 -i dynamic_pml_riker.json -f ../../AcademicPython/MPM/2dRicker/Reference/

