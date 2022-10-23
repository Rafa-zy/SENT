import torchaudio.transforms as transforms

class TimeMaskAugmentor(object):
    def __init__(self, rng, time_T, time_num_mask):
        self._tm = transforms.TimeMasking(time_T)
        self._time_num_mask = time_num_mask
        self._rng = rng
        
    def transform_audio(self, spec):
        for _ in range(self._time_num_mask):
            spec = self._tm(spec, spec.mean())

        return spec