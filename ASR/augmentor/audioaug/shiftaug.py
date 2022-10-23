class ShiftAugmentor(object):
    def __init__(self, rng, sample_rate, min_shift_ms, max_shift_ms):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._sample_rate = sample_rate
        self._rng = rng

    def transform_audio(self, audio):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        shift_samples = int(shift_ms * self._sample_rate / 1000)
        if shift_samples > 0:
            # time advance
            audio[:,:-shift_samples] = audio.clone()[:,shift_samples:]
            audio[:,-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            audio[:,-shift_samples:] = audio.clone()[:,:shift_samples]
            audio[:,:-shift_samples] = 0

        return audio