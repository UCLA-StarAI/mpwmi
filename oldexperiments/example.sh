


python3 generate-mpwmi-ijcai2020.py prova --rep 2 --vars 2 3 --clauses 1 2 --lits 2 --degree 2 --shape p
python3 light_pywmi_loop.py mpwmi --expdir results-mpwmi --dir prova --rep 2 --vars 2 3 --clauses 1 2 --lits 2 --degree 2 --problems PATH --rep 0 1
python3 light_pywmi_loop.py pa --expdir results-pa --dir prova --rep 2 --vars 2 3 --clauses 1 2 --lits 2 --degree 2 --problems PATH --rep 0 1
