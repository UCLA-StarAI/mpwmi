

EXP_NAME="bigstars"
BENCH_DIR=$EXP_NAME"/benchmark"
RES_DIR=$EXP_NAME"/results"
PLOTS_DIR=$EXP_NAME"/plots"

SHAPE="STAR"
NV="5 10 15"
NC="1 2"
NL="1 2"
DEGREE="1 2"
REPETITIONS="0 1 2 3 4"

SOLVERS="mpwmi-numeric-1 mpwmi-symbolic-1"

# GENERATE
python3 generate_benchmark.py --benchmark-dir $BENCH_DIR  --shape $SHAPE --vars $NV --clauses $NC --lits $NL --degree $DEGREE --rep $REPETITIONS

# RUN
for SOLVER in $SOLVERS
do
    python3 run_benchmark.py $SOLVER --results-dir $RES_DIR --benchmark-dir $BENCH_DIR --shape $SHAPE --vars $NV --clauses $NC --lits $NL --degree $DEGREE --rep $REPETITIONS
done

# PLOT
python3 plot_benchmark.py --solvers $SOLVERS --results-dir $RES_DIR --plots-dir $PLOTS_DIR  --shape $SHAPE --vars $NV --clauses $NC --lits $NL --degree $DEGREE --rep $REPETITIONS 

