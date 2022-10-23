import torchaudio.transforms as transforms

class FreqMaskAugmentor(object):
    def __init__(self, rng, freq_F, freq_num_mask):
        self._fm = transforms.FrequencyMasking(freq_F)
        self._freq_num_mask = freq_num_mask
        self._rng = rng
        
    def transform_audio(self, spec):
        for _ in range(self._freq_num_mask):
            spec = self._fm(spec, spec.mean())

        return spec