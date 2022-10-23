class VolumeAugmentor(object):
    def __init__(self, rng, min_gain_dBFS, max_gain_dBFS):
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS
        self._rng = rng

    def transform_audio(self, audio):
        gain = self._rng.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        audio *= 10.**(gain / 20.)
        
        return audio