import json
import random

from augmentor.audioaug.volumeaug import VolumeAugmentor
from augmentor.audioaug.speedaug import SpeedAugmentor
from augmentor.audioaug.shiftaug import ShiftAugmentor

from augmentor.specaug.freqmaskaug import FreqMaskAugmentor
from augmentor.specaug.timemaskaug import TimeMaskAugmentor
from augmentor.specaug.rectmaskaug import RectMaskAugmentor

class AugmentationPipeline(object):
    def __init__(self, sample_rate, augmentation_config):
        self._rng = random.Random()
        self._sample_rate = sample_rate
        self._audio_augmentors, self._spec_augmentors = self._aug_from_json(augmentation_config)

    def transform_audio(self, audio, mode):
        if mode == "audioaug" and self._audio_augmentors:
            for augmentor, rate in self._audio_augmentors.items():
                if self._rng.uniform(0., 1.) < rate:
                    audio = augmentor.transform_audio(audio)

        elif mode == "specaug" and self._spec_augmentors:
            for augmentor, rate in self._spec_augmentors.items():
                if self._rng.uniform(0., 1.) < rate:
                    audio = augmentor.transform_audio(audio)

        return audio

    def _aug_from_json(self, config_json):
        if config_json:
            configs = json.loads(open(config_json,'r').read())
            if configs:
                audio_augmentors = {
                    self._get_augmentor(config["type"], config["params"]):config["prob"]
                    for config in configs["audio_augmentation"]
                }

                spec_augmentors = {
                    self._get_augmentor(config["type"], config["params"]):config["prob"]
                    for config in configs["spec_augmentation"]
                }
                return audio_augmentors, spec_augmentors
        return None, None
        

    def _get_augmentor(self, augmentor_type, params):
        # audio aug
        if augmentor_type == "volume":
            return VolumeAugmentor(self._rng, **params)
        elif augmentor_type == "speed":
            return SpeedAugmentor(self._rng, **params)
        elif augmentor_type == "shift":
            return ShiftAugmentor(self._rng, self._sample_rate, **params)

        # spec aug
        elif augmentor_type == "freq_mask":
            return FreqMaskAugmentor(self._rng, **params)
        elif augmentor_type == "time_mask":
            return TimeMaskAugmentor(self._rng, **params)
        elif augmentor_type == "rect_mask":
            return RectMaskAugmentor(self._rng, **params)
        else:
            raise ValueError("Unknown augmentor type [%s]." % augmentor_type)
