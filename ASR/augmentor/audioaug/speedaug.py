import torch
import torch.nn as nn

class SpeedAugmentor(object):
    def __init__(self, rng, min_speed_rate, max_speed_rate):
        if min_speed_rate < 0.9:
            raise ValueError("Sampling speed below 0.9 can cause unnatural effects")
        if max_speed_rate > 1.1:
            raise ValueError("Sampling speed above 1.1 can cause unnatural effects")
        self._min_speed_rate = min_speed_rate
        self._max_speed_rate = max_speed_rate
        self._rng = rng

    def transform_audio(self, audio):
        speed = self._rng.uniform(self._min_speed_rate, self._max_speed_rate)
        old_length = audio.shape[1]
        new_length = int(old_length / speed)
        old_indices = torch.arange(old_length)
        new_indices = torch.linspace(0, old_length, new_length)
        audio = nn.functional.interpolate(audio.flatten()[None,None,:], size=new_length, mode='linear', align_corners=True).squeeze(0)
        
        return audio