import os.path

import numpy as np
import sys
import logging
import argparse
from collections import defaultdict
from typing import Union, Any
import warnings

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from dataset import *
from prediction import Predictor, PredictionPipeline
from config import Config

warnings.filterwarnings('ignore')


def get_logger():
    log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    root_logger = logging.getLogger('main')
    fname = os.path.join('../eval_logs.txt')
    file_handler = logging.FileHandler(os.path.join('.', fname))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel('INFO')
    return root_logger


eval_logger = get_logger()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Metric Computation Functions ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_metrics_hatecheck(true_overall: List[int], preds_overall: List[int],
                              true_per_cat: Dict[str, List[int]], preds_per_cat: Dict[str, List[int]]
                              ) -> Dict[str, Dict[str, Union[int, float]]]:
    """Compute metrics function for hatecheck.

    Computes evaluation scores for each hatecheck category and the overall scores.
    """
    # compute eval scores overall
    results = {
        'overall': {
            'acc': accuracy_score(true_overall, preds_overall),
            'f1': f1_score(true_overall, preds_overall),
            'recall': recall_score(true_overall, preds_overall),
            'precision': precision_score(true_overall, preds_overall)
        }
    }
    # compute category-wise scores
    for cat in true_per_cat:
        cat_true = true_per_cat[cat]
        cat_pred = preds_per_cat[cat]
        assert len(cat_true) == len(cat_pred)
        results[cat] = {
            'acc': accuracy_score(cat_true, cat_pred),
            'f1': f1_score(cat_true, cat_pred, zero_division=1),
            'recall': recall_score(cat_true, cat_pred, zero_division=1),
            'precision': precision_score(cat_true, cat_pred, zero_division=1),
            'num_examples': len(cat_true),
            'num_true_hate': sum(cat_true),
            'num_pred_hate': sum(cat_pred),
            'num_true_nohate': len([x for x in cat_true if x == 0]),
            'num_pred_nohate': len([x for x in cat_pred if x == 0]),
        }
    return results


def compute_metrics_default(true_overall: List[int], preds_overall: List[int], average: str = 'macro'
                            ) -> Dict[str, float]:
    """Compute metrics for all datasets except for hatecheck."""
    num_true_labels = len(set(true_overall))
    num_pred_labels = len(set(preds_overall))
    if num_pred_labels != num_pred_labels:
        eval_logger.warning('Gold and predicted labels do not have the same number of distinct labels!')
        eval_logger.warning(f'Gold labels: {set(true_overall)}')
        eval_logger.warning(f'Pred labels: {set(preds_overall)}')
        eval_logger.warning('Using num true labels.')

    if num_true_labels > 2:
        results = {
            'acc': accuracy_score(true_overall, preds_overall),
            'f1': f1_score(true_overall, preds_overall, average=average),
            'recall': recall_score(true_overall, preds_overall, average=average),
            'precision': precision_score(true_overall, preds_overall, average=average)
        }
    else:
        results = {
            'acc': accuracy_score(true_overall, preds_overall),
            'f1': f1_score(true_overall, preds_overall),
            'recall': recall_score(true_overall, preds_overall),
            'precision': precision_score(true_overall, preds_overall)
        }
    return results


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def evaluate_on_hatecheck(prediction_pipeline: PredictionPipeline, dataset: HateCheckDataset, path_out: str,
                          args: argparse.Namespace) -> None:
    true_overall = []
    preds_overall = []
    preds_per_cat = defaultdict(list)
    true_per_cat = defaultdict(list)
    raw_preds = {}
    i = 0

    eval_logger.info('Start prediction.')
    for item in tqdm(dataset, total=dataset.get_num_items()):
        pred_float = prediction_pipeline.pipeline_classify(input_text=item['text'])
        pred_int = int(round(pred_float, 0))

        # get gold label
        true_int = dataset.get_num_label(item['label'])

        # for overall results
        preds_overall.append(pred_int)
        true_overall.append(true_int)

        # for per category results
        preds_per_cat[item['category']].append(pred_int)
        true_per_cat[item['category']].append(true_int)
        raw_preds[i] = [item['category'], item['text'], true_int, pred_int]
        i += 1

    eval_logger.info('Evaluate predictions.')
    results = compute_metrics_hatecheck(true_overall=true_overall, preds_overall=preds_overall,
                                        true_per_cat=true_per_cat, preds_per_cat=preds_per_cat)
    results['cmd_args'] = args.__dict__
    results['raw_results'] = raw_preds

    eval_logger.info('Write predictions to outfile.')
    with open(path_out, 'w') as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

    results_msg = 'Results:\n---\n'
    for k, v in results['overall'].items():
        results_msg += f'- {k}: {v:.4f}\n'
    results_msg += '---'
    eval_logger.info(results_msg)


def evaluate_default(prediction_pipeline: PredictionPipeline, dataset: Dataset, path_out: str, args: argparse.Namespace
                     ) -> None:
    """The default function to evaluate a predictor on a dataset.

    Args:
        prediction_pipeline: The prediction_pipeline that is evaluated.
        dataset: The dataset used for evaluation.
        path_out: Path to file where the results are written to.
            Applies when no strategy is specified, no_nli is false and hypothesis is not used. In cases of > 2
            classes, hypothesis is always ignored and this mapping is always used.
        args: cmd-args
    """
    distinct_labels = set([item['label'] for item in dataset])
    eval_logger.info(f'Labels found in dataset: {distinct_labels}')

    true_overall = []
    preds_overall = []
    raw_preds = {}
    results = {}
    eval_logger.info('Start prediction.')
    for item in tqdm(dataset, total=dataset.get_num_items()):
        pred_float = prediction_pipeline.pipeline_classify(input_text=item['text'])
        pred_int = int(round(pred_float, 0))
        true_int = dataset.get_num_label(item['label'])
        preds_overall.append(pred_int)
        true_overall.append(true_int)
        raw_preds[item['id']] = [item['text'], true_int, pred_int, prediction_pipeline.get_partial_results()]

    eval_logger.info('Evaluate predictions.')
    results['metrics'] = compute_metrics_default(true_overall=true_overall, preds_overall=preds_overall)
    results['cmd_args'] = args.__dict__
    results['raw_results'] = raw_preds

    eval_logger.info('Write predictions to outfile.')
    with open(path_out, 'w') as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

    results_msg = 'Results:\n---\n'
    for k, v in results['metrics'].items():
        results_msg += f'- {k}: {v:.4f}\n'
    results_msg += '---'
    eval_logger.info(results_msg)


def possibly_add_filehandler(path_out: str) -> None:
    logger_path_out = os.path.join('/'.join(path_out.split('/')[:-1]), 'logs.txt')
    has_correct_handler = False
    for handler in eval_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == logger_path_out:
                has_correct_handler = True
    if not has_correct_handler:
        log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        file_handler = logging.FileHandler(logger_path_out)
        file_handler.setFormatter(log_formatter)
        eval_logger.addHandler(file_handler)
        eval_logger.info('Filehandler added.')


def load_config(fpath: str) -> Config:
    with open(fpath) as fin:
        return Config(json.load(fin))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main(args) -> None:
    config = load_config(args.path_config)
    path_out_dir = '/' + os.path.join(*config['path_out'].split('/')[:-1])
    if not os.path.exists(path_out_dir):
        os.mkdir(path_out_dir)  # create dir if it does not exist
    possibly_add_filehandler(config['path_out'])

    if os.path.exists(config['path_out']):
        raise Exception(f'Output file already exists.')

    eval_logger.info('Load test set.')
    ds_name = config['dataset']['name']
    ds_path = config['dataset']['path']
    if config['dataset'].get('task_level'):
        dataset = datasets[ds_name](ds_path, ds_name, task_level=config['dataset']['task_level'])
    else:
        dataset = datasets[ds_name](ds_path, ds_name)
    dataset.load()

    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    predictor = Predictor(model_name=config['predictor']['model'],
                          model_checkpoint=config['predictor'].get('checkpoint'), device=device, config=config)
    prediction_pipeline = PredictionPipeline(predictor=predictor, config=config)

    if ds_name == 'HateCheck':
        eval_logger.info('Use hatecheck evaluation function.')
        evaluate_on_hatecheck(prediction_pipeline=prediction_pipeline, dataset=dataset, path_out=config['path_out'],
                              args=args)
    else:
        eval_logger.info('Use default evaluation function.')
        evaluate_default(prediction_pipeline=prediction_pipeline, dataset=dataset, path_out=config['path_out'],
                         args=args)
    eval_logger.info('Finished evaluation.')


datasets = {
    'ETHOS_Binary': ETHOSBinary,
    'HateCheck': HateCheckDataset,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--path_config', help='Path to configuration file.')
    parser.add_argument('-g', '--gpu', type=int, default=-1,
                        help='GPU number to use, -1 means cpu.')
    cmd_args = parser.parse_args()
    main(cmd_args)
