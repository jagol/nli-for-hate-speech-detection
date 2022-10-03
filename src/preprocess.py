import os
import random
import re
import sys
import csv
import json
import argparse
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime

from dataset import HateCheckDataset

random.seed(5)


def get_logger() -> logging.Logger:
    log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    prepro_logger = logging.getLogger('preprocess')
    fname = os.path.join('../preprocessing_logs.txt')
    file_handler = logging.FileHandler(os.path.join('.', fname))
    file_handler.setFormatter(log_formatter)
    prepro_logger.addHandler(file_handler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(log_formatter)
    prepro_logger.addHandler(consoleHandler)
    prepro_logger.setLevel('INFO')
    return prepro_logger


def collect_corpora_to_process(args: argparse.Namespace) -> List[str]:
    if args.exclude:
        corpora = []
        excluded_by_default = ['main', 'hate_check', 'checkpoints', 'lti', 'mnli']
        for item in os.listdir(args.input):
            if not os.path.isdir(os.path.join(args.input, item)):
                continue
            if item in excluded_by_default:
                continue
            if item in args.exclude:
                continue
            corpora.append(item)
        return corpora
    elif args.include:
        return args.include
    else:
        raise Exception('Either --include or --exclude must be specified in the cmd arguments.')


class PreProcessor:

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._corpus_dir = os.path.join(args.data_dir, self._dataset_name)
        self._out_file = open(os.path.join(self._corpus_dir, f'{self._dataset_name}_preprocessed.jsonl'), 'w')

    def preprocess(self) -> None:
        self._preprocess()
        self._out_file.close()

    def _preprocess(self) -> None:
        raise NotImplementedError

    def write_to_out_file(self, item: Dict[str, Any]) -> None:
        self._out_file.write(json.dumps(item, ensure_ascii=False) + '\n')


class ETHOSBinaryPreProcessor(PreProcessor):

    def __init__(self, args: argparse.Namespace) -> None:
        self._dataset_name = 'ETHOS_Binary'
        super(ETHOSBinaryPreProcessor, self).__init__(args)
        self._fname_raw = 'Ethos_Dataset_Binary.csv'
        # self._fieldnames = ['comment', 'isHate']

    def _load_dataset(self, fname: str) -> List[Dict[str, str]]:
        dataset = []
        with open(os.path.join(self._corpus_dir, fname)) as fin:
            dreader = csv.DictReader(fin, delimiter=';')  # fieldnames=self._fieldnames,
            next(dreader)
            for i, row in enumerate(dreader):
                label_num = int(round(float(row['isHate'])))
                dataset.append({
                    'id': i,
                    'text': self._clean(row['comment']),
                    'label': row['isHate'],
                    'label_uni_bin': BinaryLabels.hate_speech if label_num else BinaryLabels.not_hate_speech,
                    'dataset': self._dataset_name,
                    'label_numeric': label_num,
                })
        return dataset

    @staticmethod
    def _clean(text: str) -> str:
        # text = re.sub(r'\s', r' ', text)
        # return re.sub(r' +', ' ', text)
        # copied from: https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/
        # 705258e118ce9611ebbd5ab2a4841b0066547293/ethos/experiments/utilities/preprocess.py#L18
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"don't", "do not ", text)
        text = re.sub(r"aren't", "are not ", text)
        text = re.sub(r"isn't", "is not ", text)
        text = re.sub(r"%", " percent ", text)
        text = re.sub(r"that's", "that is ", text)
        text = re.sub(r"doesn't", "does not ", text)
        text = re.sub(r"he's", "he is ", text)
        text = re.sub(r"she's", "she is ", text)
        text = re.sub(r"it's", "it is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text  # .decode('utf-8', 'ignore').encode("utf-8")

    def _preprocess(self) -> None:
        dataset = self._load_dataset(self._fname_raw)
        random.shuffle(dataset)
        for i, item in enumerate(dataset):
            if i < 598:
                item['split'] = Splits.train
                self.write_to_out_file(item=item)
            elif i < 798:
                item['split'] = Splits.dev
                self.write_to_out_file(item=item)
            else:
                item['split'] = Splits.test
                self.write_to_out_file(item=item)


class ETHOSMultiPreProcessor(PreProcessor):

    def __init__(self, args: argparse.Namespace) -> None:
        self._dataset_name = 'ETHOS_Multi'
        super(ETHOSMultiPreProcessor, self).__init__(args)
        self._fname_raw = 'Ethos_Dataset_Multi_Label.csv'

    def _load_dataset(self, fname: str) -> List[Dict[str, str]]:
        dataset = []
        with open(os.path.join(self._corpus_dir, fname)) as fin:
            dreader = csv.DictReader(fin, delimiter=';')  # fieldnames=self._fieldnames,
            next(dreader)
            for i, row in enumerate(dreader):
                dataset.append({
                    'id': i,
                    'text': self._clean(row['comment']),
                    'dataset': self._dataset_name,
                    'violence': float(row['violence']),
                    'directed_vs_generalized': float(row['directed_vs_generalized']),
                    'gender': float(row['gender']),
                    'race': float(row['race']),
                    'national_origin': float(row['national_origin']),
                    'disability': float(row['disability']),
                    'religion': float(row['religion']),
                    'sexual_orientation': float(row['sexual_orientation']),
                })
        return dataset

    @staticmethod
    def _clean(text: str) -> str:
        # text = re.sub(r'\s', r' ', text)
        # return re.sub(r' +', ' ', text)
        # copied from: https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/
        # 705258e118ce9611ebbd5ab2a4841b0066547293/ethos/experiments/utilities/preprocess.py#L18
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"don't", "do not ", text)
        text = re.sub(r"aren't", "are not ", text)
        text = re.sub(r"isn't", "is not ", text)
        text = re.sub(r"%", " percent ", text)
        text = re.sub(r"that's", "that is ", text)
        text = re.sub(r"doesn't", "does not ", text)
        text = re.sub(r"he's", "he is ", text)
        text = re.sub(r"she's", "she is ", text)
        text = re.sub(r"it's", "it is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text  # .decode('utf-8', 'ignore').encode("utf-8")

    def _preprocess(self) -> None:
        dataset = self._load_dataset(self._fname_raw)
        random.shuffle(dataset)
        for i, item in enumerate(dataset):
            if i < 233:
                item['split'] = Splits.train
                self.write_to_out_file(item=item)
            elif i < 333:
                item['split'] = Splits.dev
                self.write_to_out_file(item=item)
            else:
                item['split'] = Splits.test
                self.write_to_out_file(item=item)


class HateCheckPreProcessor(PreProcessor):

    def __init__(self, args: argparse.Namespace) -> None:
        self._dataset_name = 'HateCheck'
        super(HateCheckPreProcessor, self).__init__(args)
        self._fname_raw = 'hatecheck_test.csv'

    def _load_dataset(self, fname: str) -> List[Dict[str, str]]:
        dataset = []
        with open(os.path.join(self._corpus_dir, fname)) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                label_num = HateCheckDataset.labels_str_to_num[row['label_gold']]
                dataset.append({
                    'id': row[''],
                    'text': row['test_case'],
                    'label': row['label_gold'],
                    'target': row['target_ident'],
                    'label_numeric': label_num,
                    'functionality': row['functionality'],
                    'dataset': self._dataset_name,
                    'split': Splits.test,
                })
        return dataset

    def _preprocess(self) -> None:
        dataset = self._load_dataset(self._fname_raw)
        for item in dataset:
            self.write_to_out_file(item=item)


def get_timestamp() -> Tuple[str, str]:
    date, time_str = str(datetime.now()).split(' ')
    time_str = time_str.split('.')[0]
    return date, time_str


def main(args: argparse.Namespace) -> None:
    corpora = collect_corpora_to_process(args)
    args_str = json.dumps(args.__dict__, indent=4)
    logger.info(f'Command line arguments: {args_str}')
    logger.info(f'Corpora to process: {corpora}')
    for corpus in corpora:
        logger.info(f'Start processing {corpus}.')
        corpus_processor = PREPROCESSORS[corpus](args)
        corpus_processor.preprocess()
        logger.info(f'Finished processing {corpus}.')
    logger.info('Preprocessing finished.')


class BinaryLabels:
    hate_speech = 'hate-speech'
    not_hate_speech = 'not-hate-speech'


class Splits:
    train = 'train'
    dev = 'dev'
    test = 'test'


PREPROCESSORS = {
    'HateCheck': HateCheckPreProcessor,
    'ETHOS_Binary': ETHOSBinaryPreProcessor,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='data/', help='Path to input data directory.')
    parser.add_argument('-e', '--exclude', nargs='+',
                        help='All and only the corpora that not not listed and not on the default exclude '
                             'list are processed.')
    parser.add_argument('-i', '--include', nargs='+',
                        help='All and only the corpora listed are processed.')
    parser.add_argument('-t', '--task_level', required=False,
                        help='If the dataset has multiple task-level (=levels of annotation) use this arg to specify '
                             'the task level. Typically corresponds to the column name of the respective label.')
    cmd_args = parser.parse_args()
    logger = get_logger()
    main(cmd_args)
