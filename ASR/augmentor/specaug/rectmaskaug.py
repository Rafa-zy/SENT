import torch

class RectMaskAugmentor(object):
    def __init__(self, rng, rect_F, rect_T, rect_num_mask):
        self._rect_F = rect_F
        self._rect_T = rect_T
        self._rect_num_mask = rect_num_mask
        self._rng = rng
        
    def transform_audio(self, spec):
        mask = torch.zeros(spec.shape).bool()

        for _ in range(self._rect_num_mask):
            if spec.shape[0] - self._rect_F <= 0 or spec.shape[1] - self._rect_T <= 0:
                break

            rect_x = self._rng.randint(0, spec.shape[0] - self._rect_F)
            rect_y = self._rng.randint(0, spec.shape[1] - self._rect_T)

            w_x = self._rng.randint(0, self._rect_F)
            w_y = self._rng.randint(0, self._rect_T)

            mask[rect_x : rect_x + w_x, rect_y : rect_y + w_y] = True

        spec.masked_fill_(mask, spec.mean())
        del mask

        return spec