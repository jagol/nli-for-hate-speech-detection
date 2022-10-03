import argparse
import csv
import json
import logging
import os
import re
import statistics
import sys
from typing import List, Dict, Any, Union, Optional

from tqdm import tqdm

from evaluation import compute_metrics_hatecheck
from prediction import Predictor


def get_logger(args):
    log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    root_logger = logging.getLogger('main')
    fname = os.path.join(args.path_out_dir, 'logs.txt')
    file_handler = logging.FileHandler(os.path.join('.', fname))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel('INFO')
    return root_logger


def load_hatecheck(path: str) -> Dict[str, Dict[str, Any]]:
    hatecheck = {}
    with open(path) as fin:
        for line in fin:
            d = json.loads(line.strip('\n'))
            hatecheck[d['id']] = d
    return hatecheck


def compute_summary_stats(summary: Dict[str, Dict[str, Dict[str, float]]]):
    this_text_is_about_accs = []
    this_example_is_about_accs = []
    this_is_about_accs = []
    that_text_is_about_accs = []
    that_example_is_about_accs = []
    that_is_about_accs = []

    this_text_is_about_f1s = []
    this_example_is_about_f1s = []
    this_is_about_f1s = []
    that_text_is_about_f1s = []
    that_example_is_about_f1s = []
    that_is_about_f1s = []

    this_text_is_about_recalls = []
    this_example_is_about_recalls = []
    this_is_about_recalls = []
    that_text_is_about_recalls = []
    that_example_is_about_recalls = []
    that_is_about_recalls = []

    this_text_is_about_precisions = []
    this_example_is_about_precisions = []
    this_is_about_precisions = []
    that_text_is_about_precisions = []
    that_example_is_about_precisions = []
    that_is_about_precisions = []

    for test_id in summary:
        accs = []
        f1s = []
        recalls = []
        precisions = []
        for hypothesis in summary[test_id]:
            scores = summary[test_id][hypothesis]
            accs.append(scores['acc'])
            f1s.append(scores['f1'])
            recalls.append(scores['recall'])
            precisions.append(scores['precision'])
            if hypothesis.startswith('This text is about'):
                this_text_is_about_accs.append(scores['acc'])
                this_text_is_about_f1s.append(scores['f1'])
                this_text_is_about_recalls.append(scores['recall'])
                this_text_is_about_precisions.append(scores['precision'])
            elif hypothesis.startswith('This example is about'):
                this_example_is_about_accs.append(scores['acc'])
                this_example_is_about_f1s.append(scores['f1'])
                this_example_is_about_recalls.append(scores['recall'])
                this_example_is_about_precisions.append(scores['precision'])
            elif hypothesis.startswith('This is about'):
                this_is_about_accs.append(scores['acc'])
                this_is_about_f1s.append(scores['f1'])
                this_is_about_recalls.append(scores['recall'])
                this_is_about_precisions.append(scores['precision'])
            elif hypothesis.startswith('That text is about'):
                that_text_is_about_accs.append(scores['acc'])
                that_text_is_about_f1s.append(scores['f1'])
                that_text_is_about_recalls.append(scores['recall'])
                that_text_is_about_precisions.append(scores['precision'])
            elif hypothesis.startswith('That example is about'):
                that_example_is_about_accs.append(scores['acc'])
                that_example_is_about_f1s.append(scores['f1'])
                that_example_is_about_recalls.append(scores['recall'])
                that_example_is_about_precisions.append(scores['precision'])
            elif hypothesis.startswith('That is about'):
                that_is_about_accs.append(scores['acc'])
                that_is_about_f1s.append(scores['f1'])
                that_is_about_recalls.append(scores['recall'])
                that_is_about_precisions.append(scores['precision'])
            else:
                continue

        # compute mean scores for single test-id
        test_results = {'accs': [], 'f1s': [], 'recalls': [], 'precisions': []}
        for hypothesis in summary[test_id]:
            test_results['accs'].append(summary[test_id][hypothesis]['acc'])
            test_results['f1s'].append(summary[test_id][hypothesis]['f1'])
            test_results['recalls'].append(summary[test_id][hypothesis]['recall'])
            test_results['precisions'].append(summary[test_id][hypothesis]['precision'])
        test_stats = {
            'acc': statistics.mean(test_results['accs']),
            'f1': statistics.mean(test_results['f1s']),
            'recall': statistics.mean(test_results['recalls']),
            'precision': statistics.mean(test_results['precisions'])
        }
        summary[test_id]['test_stats'] = test_stats

    # compute averages
    stats = dict(
        this_text_is_about_mean_acc=sum(this_text_is_about_accs) / len(this_text_is_about_accs),
        this_example_is_about_mean_acc=sum(this_example_is_about_accs) / len(this_example_is_about_accs),
        this_is_about_mean_acc=sum(this_is_about_accs) / len(this_is_about_accs),
        that_text_is_about_mean_acc=sum(that_text_is_about_accs) / len(that_text_is_about_accs),
        that_example_is_about_mean_acc=sum(that_example_is_about_accs) / len(that_example_is_about_accs),
        that_is_about_mean_acc=sum(that_is_about_accs) / len(that_is_about_accs),
        this_text_is_about_mean_f1=sum(this_text_is_about_f1s) / len(this_text_is_about_f1s),
        this_example_is_about_mean_f1=sum(this_example_is_about_f1s) / len(this_example_is_about_f1s),
        this_is_about_mean_f1=sum(this_is_about_f1s) / len(this_is_about_f1s),
        that_text_is_about_mean_f1=sum(that_text_is_about_f1s) / len(that_text_is_about_f1s),
        that_example_is_about_mean_f1=sum(that_example_is_about_f1s) / len(that_example_is_about_f1s),
        that_is_about_mean_f1=sum(that_is_about_f1s) / len(that_is_about_f1s),
        this_text_is_about_mean_recall=sum(this_text_is_about_recalls) / len(this_text_is_about_recalls),
        this_example_is_about_mean_recall=sum(this_example_is_about_recalls) / len(this_example_is_about_recalls),
        this_is_about_mean_recall=sum(this_is_about_recalls) / len(this_is_about_recalls),
        that_text_is_about_mean_recall=sum(that_text_is_about_recalls) / len(that_text_is_about_recalls),
        that_example_is_about_mean_recall=sum(that_example_is_about_recalls) / len(that_example_is_about_recalls),
        that_is_about_mean_recall=sum(that_is_about_recalls) / len(that_is_about_recalls),
        this_text_is_about_mean_precision=sum(this_text_is_about_precisions) / len(this_text_is_about_precisions),
        this_example_is_about_mean_precision=sum(this_example_is_about_precisions) / len(
            this_example_is_about_precisions),
        this_is_about_mean_precision=sum(this_is_about_precisions) / len(this_is_about_precisions),
        that_text_is_about_mean_precision=sum(that_text_is_about_precisions) / len(that_text_is_about_precisions),
        that_example_is_about_mean_precision=sum(that_example_is_about_precisions) / len(
            that_example_is_about_precisions),
        that_is_about_mean_precision=sum(that_is_about_precisions) / len(that_is_about_precisions),
    )
    summary['stats'] = stats
    return summary


def evaluate_hypotheses_on_target(predictor: Predictor, hatecheck: Dict[str, Dict[str, Any]],
                                  hypotheses: List[str], targets: Optional[List[str]] = None,
                                  target_cats: Optional[List[str]] = None
                                  ) -> Dict[str, Dict[str, Dict[str, Union[int, float]]]]:
    results = {}
    num_hypos = len(hypotheses)
    for i, hypothesis in enumerate(hypotheses, start=1):
        logger.info(f'Start evaluating hypothesis: "{hypothesis}" [{i}/{num_hypos}]')
        pred_labels = []
        preds_per_cat = {}
        true_labels = []
        true_per_cat = {}
        for id_, item in tqdm(hatecheck.items()):
            # determine true cat and make dicts if necessary
            true_cat = item['functionality']
            if true_cat not in preds_per_cat:
                preds_per_cat[true_cat] = []
            if true_cat not in true_per_cat:
                true_per_cat[true_cat] = []

            # match = re.search(r'".+"', item['text'])
            if '[X]' in hypotheses[0]:  # for counter quote cases
                if 'counter_quote_nh' == true_cat:
                    # inner_text = match.group().strip('"')
                    outer_text = re.sub(r'".+"', '[X]', item['text'])
                    pred = predictor.nli_classify_bin(input_text=outer_text, hypothesis=hypothesis)
                else:
                    continue  # skip all test cases that are not counter quotes
            else:
                # predict and save predictions
                pred = predictor.nli_classify_bin(item['text'], hypothesis)
            pred_label = int(round(pred))
            pred_labels.append(pred_label)
            preds_per_cat[true_cat].append(pred_label)
            # determine and save true label
            if targets:
                if item['target'] in targets:
                    true_labels.append(1)
                    true_per_cat[true_cat].append(1)
                else:
                    true_labels.append(0)
                    true_per_cat[true_cat].append(0)
            elif target_cats:
                if 'counter_quote_nh' == target_cats[0]:
                    true_labels.append(0)
                    true_per_cat[true_cat].append(0)
                else:
                    if true_cat in target_cats:
                        true_labels.append(1)
                        true_per_cat[true_cat].append(1)
                    else:
                        true_labels.append(0)
                        true_per_cat[true_cat].append(0)
            else:
                raise Exception('"targets" and "target_cats" cannot both be "None".')
        logger.info('Compute metrics.')
        # remove empty cats
        key_to_del = []
        for cat in true_per_cat:
            if not true_per_cat[cat]:
                key_to_del.append(cat)
        for key in key_to_del:
            del true_per_cat[key]
        key_to_del = []
        for cat in preds_per_cat:
            if not preds_per_cat[cat]:
                key_to_del.append(cat)
        for key in key_to_del:
            del preds_per_cat[key]
        metrics = compute_metrics_hatecheck(true_labels, pred_labels, true_per_cat, preds_per_cat)
        results[hypothesis] = metrics
    return results


def hypothesis_to_score(summary) -> List[List[Union[str, float]]]:
    data = []
    for test_id in summary:
        target = test_id.split('target_')[-1]
        for hypothesis in summary[test_id]:
            data.append([
                hypothesis, target, summary[test_id][hypothesis]['acc'], summary[test_id][hypothesis]['f1'],
                summary[test_id][hypothesis]['recall'], summary[test_id][hypothesis]['precision']
            ])
    return data


def load_summary(path_out: str):
    tests_to_exclude = [
        'key_gender_target_women', 'key_gender_target_trans', 'key_queer_target_trans',
        'key_queer_target_gay', 'key_national_origin_target_black', 'key_immigrants_target_black',
        'key_race_target_immigrants', 'key_ethnicity_target_immigrants',
        'key_people_of_colour_target_immigrants', 'key_black_people_target_immigrants',
        'key_jewish_target_muslim_people', 'key_jews_target_muslim_people',
        'key_ethnicity_target_black_people_immigrants',
        # counter speech
        'key_support_target_cat_counter_quote_nh', 'key_is_for_target_cat_counter_quote_nh',
        'key_is_against_target_cat_counter_quote_nh',
        # self-directed
        'key_myself_target_cat_slur_reclaimed_nh', 'key_us_target_cat_slur_reclaimed_nh',
        # other-directed
        'key_you_target_cat_slur_reclaimed_nh', 'key_them_target_cat_slur_reclaimed_nh',
        'key_others_target_cat_slur_reclaimed_nh', 'key_other_people_target_cat_slur_reclaimed_nh',
        # dehum
        'key_animals_target_cat_derog_dehum_h', 'key_objects_target_cat_derog_dehum_h',
        'key_dehum_target_cat_derog_dehum_h', 'key_animals_target_cat_derog_impl_h',
        'key_objects_target_cat_derog_impl_h', 'key_dehum_target_cat_derog_impl_h'
    ]
    summary = {}
    for fname in os.listdir(path_out):
        if fname.startswith('key') and 'target' in fname:
            test_id = fname[:-5]
            if test_id in tests_to_exclude:
                continue
            if test_id not in summary:
                summary[test_id] = {}
            with open(os.path.join(path_out, fname)) as fin:
                fd = json.load(fin)
                for hypothesis in fd:
                    summary[test_id][hypothesis] = fd[hypothesis]['overall']
    return summary


def generate_tests() -> Dict[str, Dict[str, Union[List[str], str]]]:
    templates = [
        'This text is about {}.',
        'This is about {}.',
        'This example is about {}.',
        'That text is about {}.',
        'That example is about {}.',
        'That is about {}.',
    ]
    # gender and queer
    key_women_hypos = [t.format('women') for t in templates]
    key_gender_hypos = [t.format('gender') for t in templates]
    key_transgender_hypos = [t.format('transgender people') for t in templates]
    key_queer_hypos = [t.format('queer people') for t in templates]
    key_gay_hypos = [t.format('gay people') for t in templates]

    # race/ethnicity/national origin/immigrants
    key_race_hypos = [t.format('race') for t in templates]
    key_ethnicity_hypos = [t.format('ethnicity') for t in templates]
    key_national_origin_hypos = [t.format('national origin') for t in templates]
    key_immigrants_hypos = [t.format('immigrants') for t in templates]
    key_people_of_colour_hypos = [t.format('people of colour') for t in templates]
    key_black_people_hypos = [t.format('black people') for t in templates]

    # religion
    key_religion_hypos = [t.format('religion') for t in templates]
    key_faith_hypos = [t.format('faith') for t in templates]
    key_muslim_people_hypos = [t.format('Muslim people') for t in templates]
    key_muslims_hypos = [t.format('Muslims') for t in templates]
    # key_jewish_people_hypos = [t.format('Jewish people') for t in templates]
    # key_jews_hypos = [t.format('Jews') for t in templates]

    # disability
    key_disabled_people_hypos = [t.format('disabled people') for t in templates]
    key_disability_hypos = [t.format('disability') for t in templates]

    # counter speech / stance detection
    key_support_hypos = ['This text supports [X].', 'This example supports [X].', 'This supports [X].',
                         'That text supports [X].', 'That example supports [X].', 'That supports [X].']
    key_is_for_hypos = ['This text is for [X].', 'This example is for [X].', 'This is for [X].',
                        'That text is for [X].', 'That example is for [X].', 'That is for [X].']
    # key_is_against_hypos = ['This text is against [X].', 'This example is against [X].', 'This is against [X].',
    #                         'That text is against [X].', 'That example is against [X].', 'That is against [X].']

    # self-directed
    key_myself_hypos = [t.format('myself') for t in templates]
    key_us_hypos = [t.format('us') for t in templates]

    # other-directed
    key_you_hypos = [t.format('you') for t in templates]
    key_them_hypos = [t.format('them') for t in templates]
    key_others_hypos = [t.format('others') for t in templates]
    key_other_people_hypos = [t.format('other people') for t in templates]

    # dehumanizing comparisons
    key_animals_hypos = [t.format('animals') for t in templates]
    key_objects_hypos = [t.format('objects') for t in templates]
    key_dehum_hypos = ['This text is dehumanizing.', 'This example is dehumanizing.', 'This is dehumanizing.',
                       'This text is dehumanizing.', 'This example is dehumanizing.', 'This is dehumanizing.']

    test_dict = {
        # target women
        'key_women_target_women': {'hypotheses': key_women_hypos, 'targets': ['women']},
        'key_gender_target_women': {'hypotheses': key_gender_hypos, 'targets': ['women']},
        # target gender related
        'key_gender_target_women_trans': {'hypotheses': key_gender_hypos, 'targets': ['women', 'trans people']},
        # target transgender
        'key_trans_target_trans': {'hypotheses': key_transgender_hypos, 'targets': ['trans people']},
        'key_gender_target_trans': {'hypotheses': key_gender_hypos, 'targets': ['trans people']},
        'key_queer_target_trans': {'hypotheses': key_queer_hypos, 'targets': ['trans people']},
        # target gay
        'key_gay_target_gay': {'hypotheses': key_gay_hypos, 'targets': ['gay people']},
        'key_queer_target_gay': {'hypotheses': key_queer_hypos, 'targets': ['gay people']},
        # target queer related
        'key_queer_target_trans_gay': {'hypotheses': key_queer_hypos, 'targets': ['trans people', 'gay people']},
        # target black
        'key_race_target_black': {'hypotheses': key_race_hypos, 'targets': ['black people']},
        'key_ethnicity_target_black': {'hypotheses': key_ethnicity_hypos, 'targets': ['black people']},
        'key_national_origin_target_black': {'hypotheses': key_national_origin_hypos, 'targets': ['black people']},
        'key_immigrants_target_black': {'hypotheses': key_immigrants_hypos, 'targets': ['black people']},
        'key_people_of_colour_target_black': {'hypotheses': key_people_of_colour_hypos, 'targets': ['black people']},
        'key_black_people_target_black': {'hypotheses': key_black_people_hypos, 'targets': ['black people']},
        # target immigrant
        'key_race_target_immigrants': {'hypotheses': key_race_hypos, 'targets': ['immigrants']},
        'key_ethnicity_target_immigrants': {'hypotheses': key_ethnicity_hypos, 'targets': ['immigrants']},
        'key_national_origin_target_immigrants': {'hypotheses': key_national_origin_hypos, 'targets': ['immigrants']},
        'key_immigrants_target_immigrants': {'hypotheses': key_immigrants_hypos, 'targets': ['immigrants']},
        'key_people_of_colour_target_immigrants': {'hypotheses': key_people_of_colour_hypos, 'targets': ['immigrants']},
        'key_black_people_target_immigrants': {'hypotheses': key_black_people_hypos, 'targets': ['immigrants']},
        # target ethnicity related
        'key_ethnicity_target_black_people_immigrants': {'hypotheses': key_black_people_hypos,
                                                         'targets': ['black people', 'immigrants']},
        # target Muslim
        'key_religion_target_muslim_people': {'hypotheses': key_religion_hypos, 'targets': ['Muslims']},
        'key_faith_target_muslim_people': {'hypotheses': key_faith_hypos, 'targets': ['Muslims']},
        'key_muslim_people_target_muslim_people': {'hypotheses': key_muslim_people_hypos, 'targets': ['Muslims']},
        'key_muslims_target_muslim_people': {'hypotheses': key_muslims_hypos, 'targets': ['Muslims']},
        # 'key_jewish_target_muslim_people': {'hypotheses': key_jewish_people_hypos, 'targets': []},
        # 'key_jews_target_muslim_people': {'hypotheses': key_jews_hypos, 'targets': []},
        # target disability
        'key_disabled_people_target_disabled_people': {'hypotheses': key_disabled_people_hypos,
                                                       'targets': ['disabled people']},
        'key_disability_target_disabled_people': {'hypotheses': key_disability_hypos, 'targets': ['disabled people']},
        # counter speech
        'key_support_target_cat_counter_quote_nh': {'hypotheses': key_support_hypos,
                                                    'target_cats': ['counter_quote_nh']},
        'key_is_for_target_cat_counter_quote_nh': {'hypotheses': key_is_for_hypos, 'target_cats': ['counter_quote_nh']},
        # 'key_is_against_target_cat_counter_quote_nh': {'hypotheses': key_is_against_hypos,
        #                                                'target_cats': ['counter_quote_nh']},
        # self-directed
        'key_myself_target_cat_slur_reclaimed_nh': {'hypotheses': key_myself_hypos,
                                                    'target_cats': ['slur_reclaimed_nh']},
        'key_us_target_cat_slur_reclaimed_nh': {'hypotheses': key_us_hypos, 'target_cats': ['slur_reclaimed_nh']},
        # other-directed
        'key_you_target_cat_slur_reclaimed_nh': {'hypotheses': key_you_hypos, 'target_cats': ['slur_reclaimed_nh']},
        'key_them_target_cat_slur_reclaimed_nh': {'hypotheses': key_them_hypos, 'target_cats': ['slur_reclaimed_nh']},
        'key_others_target_cat_slur_reclaimed_nh': {'hypotheses': key_others_hypos,
                                                    'target_cats': ['slur_reclaimed_nh']},
        'key_other_people_target_cat_slur_reclaimed_nh': {'hypotheses': key_other_people_hypos,
                                                          'target_cats': ['slur_reclaimed_nh']},
        # dehumanizing comparisons
        'key_animals_target_cat_derog_dehum_h_derog_impl_h': {'hypotheses': key_animals_hypos,
                                                              'target_cats': ['derog_dehum_h', 'derog_impl_h']},
        'key_objects_target_cat_derog_dehum_h_derog_impl_h': {'hypotheses': key_objects_hypos,
                                                              'target_cats': ['derog_dehum_h', 'derog_impl_h']},
        'key_dehum_target_cat_derog_dehum_h_derog_impl_h': {'hypotheses': key_dehum_hypos,
                                                            'target_cats': ['derog_dehum_h', 'derog_impl_h']},
    }
    return {'key_support_target_cat_counter_quote_nh': {'hypotheses': key_support_hypos,
                                                        'target_cats': ['counter_quote_nh']},
            'key_is_for_target_cat_counter_quote_nh': {'hypotheses': key_is_for_hypos,
                                                       'target_cats': ['counter_quote_nh']}}


def generate_stance_detect_hypos_rank(path_out_dir: str) -> List[List[Union[str, float]]]:
    fnames = ['key_support_target_cat_counter_quote_nh.json', 'key_is_for_target_cat_counter_quote_nh.json']
    stance_hypo_rank = []
    for fname in fnames:
        with open(os.path.join(path_out_dir, fname)) as fin:
            file_dict = json.load(fin)
            for hypothesis in file_dict:
                stance_hypo_rank.append([
                    hypothesis,
                    file_dict[hypothesis]['counter_quote_nh']['acc'],
                    file_dict[hypothesis]['counter_quote_nh']['f1'],
                    file_dict[hypothesis]['counter_quote_nh']['recall'],
                    file_dict[hypothesis]['counter_quote_nh']['precision'],
                ])
    return stance_hypo_rank


def generate_self_directed_rank(path_out_dir: str) -> List[List[Union[str, float]]]:
    fnames = ['key_myself_target_cat_slur_reclaimed_nh.json', 'key_us_target_cat_slur_reclaimed_nh.json']
    self_directed_hypo_rank = []
    for fname in fnames:
        with open(os.path.join(path_out_dir, fname)) as fin:
            file_dict = json.load(fin)
            for hypothesis in file_dict:
                self_directed_hypo_rank.append([
                    hypothesis,
                    file_dict[hypothesis]['overall']['acc'],
                    file_dict[hypothesis]['overall']['f1'],
                    file_dict[hypothesis]['overall']['recall'],
                    file_dict[hypothesis]['overall']['precision'],
                ])
    return self_directed_hypo_rank


def main(args: argparse.Namespace) -> None:
    if not args.only_summary:  # check if only summary should be generated
        hatecheck = load_hatecheck(args.hatecheck)
        device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
        predictor = Predictor(args.model, args.checkpoint, device, config={'predictor': []})

        test_dict = generate_tests()
        num_tests = len(test_dict)

        for i, (test_id, test_dict) in enumerate(test_dict.items(), start=1):
            fpath_out = os.path.join(args.path_out_dir, test_id + '.json')
            logger.info(f'Start test: {test_id} [{i}/{num_tests}]')
            if 'targets' in test_dict:
                results = evaluate_hypotheses_on_target(predictor=predictor, hatecheck=hatecheck,
                                                        hypotheses=test_dict['hypotheses'],
                                                        targets=test_dict['targets'])
            elif 'target_cats' in test_dict:
                results = evaluate_hypotheses_on_target(predictor=predictor, hatecheck=hatecheck,
                                                        hypotheses=test_dict['hypotheses'],
                                                        target_cats=test_dict['target_cats'])
            else:
                raise Exception(f'Test "{test_id}" contains no "target" or "target_cat".')
            logger.info(f"Write results to file: {fpath_out}")
            with open(fpath_out, 'w') as fout:
                json.dump(results, fout, indent=4)

    summary = load_summary(args.path_out_dir)
    hypotheses_scores = hypothesis_to_score(summary)
    # summary_w_stats = compute_summary_stats(summary)
    stance_detect_hypos_rank = generate_stance_detect_hypos_rank(args.path_out_dir)
    # self_directed_hypos_rank = generate_self_directed_rank(args.path_out_dir)

    # write to file
    path_out_sum = os.path.join(args.path_out_dir, 'summary.json')
    path_out_target_hypo_rank = os.path.join(args.path_out_dir, 'target_hypo_ranked.csv')
    path_out_stance_detect_rank = os.path.join(args.path_out_dir, 'stance_detect_hypo_rank.csv')
    path_out_self_directed_rank = os.path.join(args.path_out_dir, 'self_directed_hypo_rank.csv')
    # logger.info(f'Write summary to file: {path_out_sum}')
    # with open(path_out_sum, 'w') as fout:
    #     json.dump(summary_w_stats, fout, indent=4)

    # logger.info(f'Write target hypothesis ranking to file: {path_out_target_hypo_rank}')
    # with open(path_out_target_hypo_rank, 'w') as fout:
    #     writer = csv.writer(fout)
    #     writer.writerow(['hypothesis', 'target', 'acc', 'f1', 'recall', 'precision'])
    #     for hr in hypotheses_scores:
    #         writer.writerow(hr)

    logger.info(f'Write stance detection ranking to file: {path_out_stance_detect_rank}')
    with open(path_out_stance_detect_rank, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['hypothesis', 'acc', 'f1', 'recall', 'precision'])
        for item in stance_detect_hypos_rank:
            writer.writerow(item)

    # logger.info(f'Write self-directed ranking to file: {path_out_self_directed_rank}')
    # with open(path_out_self_directed_rank, 'w') as fout:
    #     writer = csv.writer(fout)
    #     writer.writerow(['hypothesis', 'acc', 'f1', 'recall', 'precision'])
    #     for item in self_directed_hypos_rank:
    #         writer.writerow(item)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--hatecheck', help='Path to hatecheck corpus in jsonl.')
    parser.add_argument('-o', '--path_out_dir', help='Path to output directory where all output-files are written to.')
    parser.add_argument('-m', '--model', help='The model type to be used. HugginFace identifier.')
    parser.add_argument('-c', '--checkpoint', default=None, help='Path to checkpoint to be loaded. Optional.')
    parser.add_argument('-s', '--only_summary', action='store_true', help='If set, only compute the overall summary.')
    parser.add_argument('-g', '--gpu', type=int, help='GPU to be used.')
    parser.add_argument('-k', '--hate_hypo_key', default='this-text-contains-hate-speech',
                        help='Choose the main hypothesis that is used for detecting presence of hate speech.')
    cmd_args = parser.parse_args()
    logger = get_logger(cmd_args)
    main(cmd_args)
