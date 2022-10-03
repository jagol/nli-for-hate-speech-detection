import argparse
import json
import os
from statistics import mean
from typing import Dict, List, Tuple


def collect_relevant_ETHOS_accs(path_dir: str) -> Dict[str, float]:
    rel_accs = {
        'only_HSCatcher': None,
        'FBT_tg': None,
        'FBT_tc': None,
        'FC': None,
        'FRS': None,
        'CDC': None,
        'FBT_tc_FC': None,
        'FBT_tc_FRS': None,
        'FC_FRS': None,
        'FBT_tc_FC_FRS': None,
        'CDC_FBT_tc_FC_FRS': None
    }
    for fname in os.listdir(path_dir):
        config_name = fname[:-22]
        if config_name in rel_accs:
            with open(os.path.join(path_dir, fname)) as fin:
                d = json.load(fin)
                acc = d['metrics']['acc']
                rel_accs[config_name] = acc
    return rel_accs


def compute_diffs_accs(accs: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    accs_diccs = {}
    base_acc = accs['only_HSCatcher']
    for cname in accs:
        accs_diccs[cname] = (accs[cname], accs[cname] - base_acc)
    return accs_diccs


def to_latex_table(accs_diffs: Dict[str, Tuple[float, float]]) -> str:
    table = f"""\hline
strategies & accuracy (\%) & $\Delta$ \\\\
\hdashline
(ETHOS) SVM & 66.4 & - \\\\
(ETHOS) BERT & 80.0 & - \\\\
(ETHOS) DistilBERT & 80.4 & - \\\\
\hdashline
``That contains hate speech.'' & {100*accs_diffs['only_HSCatcher'][0]:.1f} & {100*accs_diffs['only_HSCatcher'][1]:+.1f} \\\\
\hdashline
FBT (TG) & {100*accs_diffs['FBT_tg'][0]:.1f} & {100*accs_diffs['FBT_tg'][1]:+.1f} \\\\
FBT (TC) & {100*accs_diffs['FBT_tc'][0]:.1f} & {100*accs_diffs['FBT_tc'][1]:+.1f} \\\\
FCS & {100*accs_diffs['FC'][0]:.1f} & {100*accs_diffs['FC'][1]:+.1f} \\\\
FRS & {100*accs_diffs['FRS'][0]:.1f} & {100*accs_diffs['FRS'][1]:+.1f} \\\\
CDC (TC) & {100*accs_diffs['CDC'][0]:.1f} & {100*accs_diffs['CDC'][1]:+.1f} \\\\
\hdashline
FBT (TC) + FCS & {100*accs_diffs['FBT_tc_FC'][0]:.1f} & {100*accs_diffs['FBT_tc_FC'][1]:+.1f} \\\\
FBT (TC) + FRS & {100*accs_diffs['FBT_tc_FRS'][0]:.1f} & {100*accs_diffs['FBT_tc_FRS'][1]:+.1f} \\\\
FCS + FRS & {100*accs_diffs['FC_FRS'][0]:.1f} & {100*accs_diffs['FC_FRS'][1]:+.1f} \\\\
FBT (TC) + FCS + FRS & {100*accs_diffs['FBT_tc_FC_FRS'][0]:.1f} & {100*accs_diffs['FBT_tc_FC_FRS'][1]:+.1f} \\\\
CDC (TC) + FBT (TC) + FCS + FRS & {100*accs_diffs['CDC_FBT_tc_FC_FRS'][0]:.1f} & {100*accs_diffs['CDC_FBT_tc_FC_FRS'][1]:+.1f} \\\\
\hline"""
    return table


def main(args: argparse.Namespace) -> None:
    accs = collect_relevant_ETHOS_accs(args.dir)
    diffs_accs = compute_diffs_accs(accs)
    latex_table = to_latex_table(diffs_accs)
    print(latex_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='Path to directory containins ETHOS results files.')
    cmd_args = parser.parse_args()
    main(cmd_args)