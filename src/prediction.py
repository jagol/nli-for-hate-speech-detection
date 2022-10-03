import json
import logging
import os
import re
import statistics
import sys
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config


def get_logger():
    log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    root_logger = logging.getLogger('prediction')
    fname = os.path.join('../pred_logs.txt')
    file_handler = logging.FileHandler(os.path.join('.', fname))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel('INFO')
    return root_logger


pred_logger = get_logger()


class Predictor:

    def __init__(self, model_name: str, model_checkpoint: Optional[str], device: str, config: Config) -> None:
        pred_logger.info('Initialize Predictor.')
        self._model_name = model_name
        self._model_checkpoint = model_checkpoint
        self._config = config
        if 'label_mapping' in config['predictor']:
            self._entail_idx = config['predictor']['label_mapping']['entailment']
            self._contra_idx = config['predictor']['label_mapping']['contradiction']
        else:
            # default mapping of bart-large-mnli
            self._entail_idx = 2
            self._contra_idx = 0
        self._device = device
        pred_logger.info('Load tokenizer.')
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        pred_logger.info(f'Load model {model_name} from checkpoint {model_checkpoint}.')
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint if model_checkpoint else model_name
        )
        pred_logger.info(f'Move model to device: {self._device}')
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def nli_classify_bin(self, input_text: str, hypothesis: str) -> float:
        """Do binary NLI classification.

        Args:
            input_text: text to be classified/premise
            hypothesis: one hypothesis
        Return:
            prob_entail: The probability of entailment, meaning
                the prob. that the hypothesis is true.
        """
        encoded_input = self._tokenizer.encode(
            input_text,
            hypothesis,
            return_tensors='pt',
            truncation='only_first'
        )
        logits = self._model(encoded_input.to(self._device))[0]
        contradiction_entail_logits = logits[0, [self._contra_idx, self._entail_idx]]  # 0-indexing assumes exactly one
                                                                                       # example
        probs = contradiction_entail_logits.softmax(dim=0)
        # dim=0 only because already extracted one example with a batch it would be dim=1.
        prob_entail = probs[1].item()
        return prob_entail

    @torch.no_grad()
    def nli_classify_multi(self, input_text: str, hypotheses: List[str]) -> List[float]:
        """Do NLI classification for more than 2 classes.

        This implies 2 or more hypotheses

        Args:
            input_text: text to be classified/premise
            hypotheses: hypotheses
        Return:
            probs: probabilities corresponding to the hypotheses (same order)
        """
        probs_raw = [self.nli_classify_bin(input_text, hypothesis) for hypothesis in hypotheses]
        # return torch.nn.functional.softmax(torch.FloatTensor(probs_raw), dim=0).tolist()
        return probs_raw

    @torch.no_grad()
    def classify_no_nli(self, input_text) -> List[float]:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
        Return:
            List of class probabilities.
        """
        encoded_input = self._tokenizer.encode(input_text, return_tensors='pt', truncation='only_first')
        logits = self._model(encoded_input.to(self._device))[0]
        prob_distr = torch.softmax(logits, dim=1)
        return prob_distr[0].tolist()


class PipelineSection:
    """A filter that returning a float x: 0 <= x <= 1.

    1 is a pass, 0 is a fail.
    """

    def __init__(self, predictor: Optional[Predictor], config: Config):
        self._predictor = predictor
        self._config = config

    def apply(self, input_text: str) -> float:
        raise NotImplementedError


class TargetFilter(PipelineSection):
    """Filter to test if a target group is mentioned in the input text.

    Predict for all targets the probability that they are being mentioned.
    Return the highest predicted probability.
    """

    def __init__(self, predictor: Predictor, config: Config) -> None:
        super().__init__(predictor, config)
        self._target_hypos = self._config.get_hypos_for_section('filters', self.__class__.__name__)
        pred_logger.info(f'Initialized {self.__class__.__name__} with the following target hypotheses: '
                         f'{json.dumps(self._target_hypos, indent=4)}')

    def apply(self, input_text: str, ) -> float:
        results = []
        for target, target_hypo in self._target_hypos.items():
            results.append(self._predictor.nli_classify_bin(input_text=input_text, hypothesis=target_hypo))
        return max(results)


class TargetListFilter(PipelineSection):

    def __init__(self, config: Config) -> None:
        super(TargetListFilter, self).__init__(predictor=None, config=config)
        self._config = config
        self._path_target_words = config['prediction_pipeline']['filters']['TargetListFilter']['path']
        with open(self._path_target_words) as fin:
            self._target_words = [line.strip('\n') for line in fin.readlines()]

    def apply(self, input_text: str) -> float:
        for tw in self._target_words:
            if tw in input_text:
                return 1.0
        return 0.0


class CDCListCatcher(PipelineSection):

    def __init__(self, config: Config) -> None:
        super(CDCListCatcher, self).__init__(predictor=None, config=config)
        self._config = config
        self._path_target_words = config['prediction_pipeline']['filters']['CDCListFilter']['path']
        with open(self._path_target_words) as fin:
            lines = fin.readlines()
            self._animals = [line.strip('\n') for line in lines if line or not line.startswith('Source:')]

    def apply(self, input_text: str) -> float:
        for tw in self._animals:
            if tw in input_text:
                return 1.0
        return 0.0


class RecSlurFilter(PipelineSection):
    """Filter that tests for a slurs being reclaimed by using self-directedness as a proxy.

    Return the probability that the input text is self-directed (and thus not hateful if it contains a slur).
    """

    def __init__(self, predictor: Predictor, config: Config):
        super().__init__(predictor, config)
        self._hypotheses = self._config.get_hypos_for_section('filters', self.__class__.__name__)
        self._about_others_thresh = self._config['prediction_pipeline']['filters']['RecSlurFilter']['thresholds']['about_others']
        self._neg_senti_thresh = self._config['prediction_pipeline']['filters']['RecSlurFilter']['thresholds']['neg_senti']
        pred_logger.info(f'Initialized {self.__class__.__name__} with the following hypothesis: '
                         f'{self._hypotheses}')

    # def _get_neg_senti_score(self, input_text: str) -> float:
    #     return self._predictor.nli_classify_bin(input_text=input_text, hypothesis=self._hypotheses['neg-senti'])

    def _get_about_others_score(self, input_text: str) -> float:
        return 1 - self._predictor.nli_classify_bin(input_text=input_text, hypothesis=self._hypotheses['myself'])

    def apply(self, input_text: str) -> float:
        about_others_score = self._get_about_others_score(input_text=input_text)
        # neg_senti_score = self._get_neg_senti_score(input_text=input_text)
        # if about_others_score > self._about_others_thresh and neg_senti_score >= self._neg_senti_thresh:
        #     return 1.0
        # return 0.0
        return about_others_score


class CSFilter(PipelineSection):
    """Filter aimed at detecting counter speech.

    Steps:
        1. Detection of quote.
        2. Predict if quote is hate speech.
        3. If yes, predict stance of outer text towards quote. (An against-stance implies counter-speech).

    Uses the hate speech hypothesis from HSCatcher (relies on a HSCatcher existing in the pipeline).

    If hate speech is quoted return the probability that the outer text's stance is against it.
    Otherwise, return 1 (filter passed).
    """

    def __init__(self, predictor: Predictor, config: Config):
        super().__init__(predictor, config)
        self._stance_support_hypo = self._config.get_hypos_for_section('filters', self.__class__.__name__)
        self._hate_hypo = self._config.get_hypos_for_section('catchers', "HSCatcher")
        self._hate_threshold = config['prediction_pipeline']['filters'][self.__class__.__name__].get(
            'hate_threshold', 0.5)
        pred_logger.info(f'Initialized {self.__class__.__name__} with the following hypotheses:\n'
                         f'stance: {self._stance_support_hypo}\nhate: {self._hate_hypo}')
        if config['prediction_pipeline']['filters'][self.__class__.__name__].get('TargetFilter'):
            self._target_filter = True
            self._target_hypos = config.get_from_key_list(config['prediction_pipeline']['filters']
                                                          [self.__class__.__name__]['TargetFilter']['hypotheses_keys'])
        else:
            self._target_filter = False

    def apply(self, input_text: str) -> float:
        match = re.search(r'".+"', input_text)
        if match:
            inner_text = match.group().strip('"')
            outer_text = re.sub(r'".+"', '[X]', input_text)
            i_hate_result = self._predictor.nli_classify_bin(input_text=inner_text, hypothesis=self._hate_hypo)
            # o_hate_result = self._predictor.nli_classify_bin(input_text=outer_text, hypothesis=self._hate_hypo)
            if i_hate_result >= self._hate_threshold:
                o_stance = self._predictor.nli_classify_bin(input_text=outer_text, hypothesis=self._stance_support_hypo)
                return o_stance
            else:
                return 0
            # if self._target_filter:
            #     target_results = []
            #     for target, target_hypo in self._target_hypos.items():
            #         target_results.append(self._predictor.nli_classify_bin(input_text=input_text, hypothesis=target_hypo))
            #     o_hate_result = max(target_results)
            # if i_hate_result < self._hate_threshold and o_hate_result < self._hate_threshold:
            #     return statistics.mean([i_hate_result, o_hate_result])
            # elif i_hate_result >= self._hate_threshold > o_hate_result:
            #     o_stance = self._predictor.nli_classify_bin(input_text=outer_text, hypothesis=self._stance_support_hypo)
            #     return o_stance
            # elif i_hate_result < self._hate_threshold <= o_hate_result:
            #     return o_hate_result
            # else:
            #     return statistics.mean([i_hate_result, o_hate_result])
        return 1


class DehumCompCatcher(PipelineSection):

    def __init__(self, predictor: Predictor, config: Config) -> None:
        super().__init__(predictor, config)
        self._dehum_catch_hypos = self._config.get_hypos_for_section('catchers', self.__class__.__name__)
        target_hypos = self._config.get_hypos_for_section('catchers', self.__class__.__name__)
        if self._config['prediction_pipeline']['catchers']['DehumCompCatcher']['target_type'] == 'groups':
            self._target_hypos = target_hypos['groups']
        elif self._config['prediction_pipeline']['catchers']['DehumCompCatcher']['target_type'] == 'characteristics':
            self._target_hypos = target_hypos['characteristics']
        self._target_group_thresh = self._config['prediction_pipeline']['catchers']['DehumCompCatcher']['thresholds']['target_group']
        self._neg_senti_score_thresh = self._config['prediction_pipeline']['catchers']['DehumCompCatcher']['thresholds']['neg_senti_score']
        self._neg_animals_thresh = self._config['prediction_pipeline']['catchers']['DehumCompCatcher']['thresholds']['neg_animals']
        pred_logger.info(f'Initialized {self.__class__.__name__} with the following dehum hypotheses:\n'
                         f'dehum: {self._dehum_catch_hypos}')
        pred_logger.info(f'Initialized {self.__class__.__name__} with the following target hypotheses:\n'
                         f'dehum: {self._target_hypos}')

    def _apply_target_filter(self, input_text: str, ) -> float:
        results = []
        for target, target_hypo in self._target_hypos.items():
            results.append(self._predictor.nli_classify_bin(input_text=input_text, hypothesis=target_hypo))
        return max(results)

    def _get_neg_senti_score(self, input_text: str) -> float:
        return self._predictor.nli_classify_bin(input_text=input_text, hypothesis=self._dehum_catch_hypos['neg-senti'])

    def _get_neg_animals_score(self, input_text: str) -> float:
        hypotheses = [
            self._dehum_catch_hypos['insects'],
            self._dehum_catch_hypos['rats'],
            self._dehum_catch_hypos['apes'],
            self._dehum_catch_hypos['primates'],
            self._dehum_catch_hypos['plague'],
        ]
        scores = [self._predictor.nli_classify_bin(input_text=input_text, hypothesis=hypo) for hypo in hypotheses]
        return max(scores)

    def apply(self, input_text: str) -> float:
        target_group_score = self._apply_target_filter(input_text=input_text)
        if target_group_score > self._target_group_thresh:
            neg_senti_score = self._get_neg_senti_score(input_text=input_text)
            if neg_senti_score > self._neg_senti_score_thresh:
                dehum_animals_score = self._get_neg_animals_score(input_text=input_text)
                if dehum_animals_score >= self._neg_animals_thresh:
                    return statistics.mean([target_group_score, neg_senti_score, dehum_animals_score])
        return 0.0


class HSCatcher(PipelineSection):

    def __init__(self, predictor: Predictor, config: Config):
        super().__init__(predictor, config)
        hate_speech_keys = config['prediction_pipeline']['catchers'][self.__class__.__name__]['hypotheses_keys']
        self._hate_speech_hypo = config.get_from_key_list(hate_speech_keys)

    def apply(self, input_text: str) -> float:
        return self._predictor.nli_classify_bin(input_text=input_text, hypothesis=self._hate_speech_hypo)


class HSCatcherNoNLI(PipelineSection):

    def __init__(self, predictor: Predictor, config: Config):
        super().__init__(predictor, config)

    def apply(self, input_text: str) -> float:
        return self._predictor.classify_no_nli(input_text=input_text)[1]


class PredictionPipeline:

    _catcher_clss = {catcher_cls.__name__: catcher_cls for catcher_cls in [HSCatcher, DehumCompCatcher, HSCatcherNoNLI]}

    _filter_clss = {filter_cls.__name__: filter_cls for filter_cls in [TargetFilter, DehumCompCatcher, RecSlurFilter,
                                                                       CSFilter]}

    def __init__(self, predictor: Predictor, config: Config):
        """

        Args:
            predictor: Predictor
            config:
        """
        self._predictor = predictor
        self._config = config
        self._catch_threshold = self._config['prediction_pipeline']['catch_threshold']
        if self._config['prediction_pipeline'].get('no_nli', False):
            self._pred_method = self.no_nli_pipeline_classify
        else:
            self._pred_method = self.nli_pipeline_classify
        pred_logger.info(f'Prediction method set to: {self._pred_method}')
        self._catcher_insts = []
        self._filter_insts = []
        for id_ in config['prediction_pipeline']['catchers']:
            self._catcher_insts.append(self._catcher_clss[id_](predictor, config))
        for id_ in config['prediction_pipeline'].get('filters', []):
            self._filter_insts.append(self._filter_clss[id_](predictor, config))
        self._catcher_results = {}
        self._filter_results = {}

        self._comb_strats = {
            'only_HSCatcher': self._only_hscatcher,
            'max_catch_min_filter': self._max_catch_min_filter,
            'indiv_threshold': self._indiv_threshold,
        }
        self._comb_strat = self._config['prediction_pipeline']['comb_strat']
        pred_logger.info(f'Initiated prediction pipeline with:\ncatchers: '
                         f'{[c.__class__.__name__ for c in self._catcher_insts]}\nfilters:'
                         f'{[f.__class__.__name__ for f in self._filter_insts]}\n'
                         f'comb_strat: {self._comb_strat}')

    def pipeline_classify(self, input_text: str) -> float:
        return self._pred_method(input_text)

    def no_nli_pipeline_classify(self, input_text: str) -> float:
        return self._catcher_insts[0].apply(input_text)

    def nli_pipeline_classify(self, input_text: str) -> float:
        """Apply a pipeline of filters and catchers."""
        for catcher in self._catcher_insts:
            self._catcher_results[catcher.__class__.__name__] = catcher.apply(input_text=input_text)
        for filter_ in self._filter_insts:
            part_result = filter_.apply(input_text=input_text)
            self._filter_results[filter_.__class__.__name__] = part_result
        return self._comb_strats[self._config['prediction_pipeline']['comb_strat']]()

    def _only_hscatcher(self) -> float:
        return self._catcher_results['HSCatcher']

    def _indiv_threshold(self) -> float:
        for filter_name, value in self._filter_results.items():
            threshold = self._config['prediction_pipeline']['filters'][filter_name]['threshold']
            if value <= threshold:
                return 0.0
        # hs_prob = 0.0
        # for catcher_name, value in self._catcher_results.items():
        #     threshold = self._config['prediction_pipeline']['catchers'][catcher_name]['threshold']
        #     if value >= threshold and value >= hs_prob:
        #         return
        # return hs_prob
        return max(self._catcher_results.values())

    def _max_catch_min_filter(self) -> float:
        if max(self._catcher_results.values()) >= self._catch_threshold:
            if len(self._filter_results.values()) > 0:
                return min(self._filter_results.values())
            else:
                return max(self._catcher_results.values())
        else:
            return max(self._catcher_results.values())

    def _hsc_min_filter(self) -> float:
        if max(self._catcher_results['HSC']) >= self._catch_threshold:
            return min(self._filter_results.values())
        else:
            return max(self._catcher_results.values())

    def _hsc_avg_filter(self) -> float:
        if max(self._catcher_results['HSC']) >= self._catch_threshold:
            return statistics.mean(self._filter_results.values())
        else:
            return max(self._catcher_results.values())

    def get_partial_results(self) -> Dict[str, Dict[str, float]]:
        return {'catchers': self._catcher_results, 'filters': self._filter_results}
