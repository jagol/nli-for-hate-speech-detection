#!/bin/bash
# $1: path to configs directory, str
# $2: path to data directory, str
# $3: gpu to use, int
python3 src/preprocess.py -i ETHOS_Binary -d ${2}
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FBT_tc_FC_FRS_that_contains_HS.json -g $3
# python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tge_FC_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tg_FC_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FBT_tge_FC_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tc_FC_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tge_FC_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tg_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FBT_tg_FC_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tc_FC_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tge_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tg_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FC_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tc_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tge_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FC_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tc_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FBT_tg_FC_FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FRS_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FBT_tg_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FBT_tge_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/CDC_FBT_tc_that_contains_HS.json -g $3
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_ETHOS/FC_FRS_that_contains_HS.json -g $3

# python3 src/evaluation.py -c /srv/scratch0/jgoldz/nli-for_hate-speech-detection/configs/zero-shot/compare_strategies_on_ETHOS/CDC_FBT_tc_FC_FRS_that_contains_HS.json -g 6