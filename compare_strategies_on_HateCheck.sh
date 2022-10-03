#!/bin/bash
# $1: path to configs directory, str
# $2: gpu to use, int
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FBT_tc_FC_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tge_FC_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tg_FC_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FBT_tge_FC_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tc_FC_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tge_FC_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tg_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FBT_tg_FC_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tc_FC_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tge_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tg_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FC_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tc_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tge_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FC_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tc_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FBT_tg_FC_FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FRS_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FBT_tg_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FBT_tge_that_contains_HS.json -g $2
# python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/CDC_FBT_tc_that_contains_HS.json -g $2
python3 src/evaluation.py -c ${1}/zero-shot/compare_strategies_on_HateCheck/FC_FRS_that_contains_HS.json -g $2
