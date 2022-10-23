import math
import random
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torchaudio
import torchaudio.transforms as transforms
from torchaudio.compliance import kaldi
from util import read_manifest
from augmentor.augmentation import AugmentationPipeline
import numpy as np

class DatasetXL(Dataset):
    def __init__(self, manifest, vocab_dict, vocab_list,
                sample_rate=16000, preemphasis=0.97, num_concat=5,
                n_fft=320, win_len=320, hop_len=160, n_mels=64, augmentation_config=None):

        self._manifest = manifest
        self._vocab_dict = vocab_dict
        self._vocab_list = vocab_list
        self._sample_rate = sample_rate
        self._preemphasis = preemphasis
        self._featurizer = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_len, hop_length=hop_len, n_mels=n_mels)
        self._num_concat = num_concat
        self._augmentation_pipeline = AugmentationPipeline(sample_rate=sample_rate, augmentation_config=augmentation_config)
        
    def __getitem__(self, index):
        line = self._manifest[index]

        label = self._token_to_idx(line['text'].strip())
        try:
            label_ori = self._token_to_idx(line['text_ori'].strip())
        except:
            label_ori = self._token_to_idx('是') # for data without orilabel
            # raise ValueError(f"no index error,check raw json data file, label is {line['text'].strip()}")
        try:
            ind = int(line['index'])
        except:
            raise ValueError("no index error,check raw json data file")
        # read, mono and resample
        try:
            audio, sample_rate = torchaudio.load(line['audio_filepath'])
        except:
            print(f"wrong file: {line['audio_filepath']}")
            return
        audio = torch.mean(audio, dim=0, keepdim=True) 
        audio = kaldi.resample_waveform(audio, sample_rate, self._sample_rate) 
        audio = self._augmentation_pipeline.transform_audio(audio, mode='audioaug')
        audio = audio.squeeze(0) # eliminate channel dim

        # pre-emphasis
        if self._preemphasis:
            audio = torch.cat((audio[0:1], audio[1:] - self._preemphasis * audio[:-1]))
        
        # featurize
        spec = self._featurizer(audio)
        spec = torch.log(spec + torch.finfo(spec.dtype).eps)
        del audio
        spec = self._augmentation_pipeline.transform_audio(spec, mode='specaug')
        spec = spec.T # dim * time -> time * dim
        
        if self._num_concat > 1:
            spec = self._merge_spec(spec)
        
        return spec, label, ind,label_ori

    def _merge_spec(self, spec):
        time, dim = spec.shape[0], spec.shape[1]
        padding_distance = int(math.ceil(time / self._num_concat) * self._num_concat) - time
        padding = torch.zeros(padding_distance, dim).type_as(spec)
        spec = torch.cat((spec, padding), dim=0).reshape(-1, dim * self._num_concat)

        return spec

    def _token_to_idx(self, tokens):
        label = torch.tensor([self._vocab_dict[token] for token in tokens if token in self._vocab_dict], dtype=torch.long)
        eos = torch.zeros(1).type_as(label)
        label = torch.cat((eos, label, eos))

        return label

    def __len__(self):
        return len(self._manifest)

    def get_vocab_size(self):
        return len(self._vocab_list)

class BatchSamplerXL(Sampler):
    def __init__(self, data_source, durations, mode, max_bsz, max_batch_duration, shuffle=True):
        self._data_source = data_source
        self._durations = durations
        self._mode = mode
        self._max_bsz = max_bsz
        self._max_batch_duration = max_batch_duration
        self.shuffle = shuffle
        self.first_shuffle = False
        self._num_groups = 10
        
    def __iter__(self):
        # data = list(zip(self._data_source, self._durations))
        data_raw = np.array(list(zip(self._data_source, self._durations)))
        # if self._mode == 'train' and (not self.first_shuffle or self.shuffle):
        #     shift = min(len(self._data_source)-1, random.randint(0, self._max_bsz))
        #     data = data[shift:]
        if self._mode == "train" and self.shuffle:
            data_groups = np.array_split(data_raw, self._num_groups)
        else:
            data_groups = np.array_split(data_raw, 1)
        end_of_batch = []
        batches = []
        cur_eob = []
        cur_batch = []
        cur_duration = 0.0
        for dg in data_groups:
            group = dg.tolist()
            group.sort(key=lambda x:x[1], reverse=False)
            for line in dg:
                duration = line[1]
                if cur_duration + duration > self._max_batch_duration and len(cur_batch) > 0 or len(cur_batch) == self._max_bsz:
                    cur_eob[-1] = True
                    end_of_batch.append(cur_eob)
                    batches.append(cur_batch)
                    cur_batch = []
                    cur_eob = []
                    cur_duration = 0.0
                cur_batch.append(line)
                cur_eob.append(False)
                cur_duration += duration
        if len(cur_batch) > 0:
            cur_eob[-1] = True
            end_of_batch.append(cur_eob)
            batches.append(cur_batch)
        del cur_batch, cur_duration, cur_eob
        batch_with_eob = list(zip(batches, end_of_batch))
        if self._mode == 'train' and (not self.first_shuffle or self.shuffle):
            random.shuffle(batch_with_eob)
            self.first_shuffle = True
        data = []
        end_of_batch = []
        data_, end_of_batch_ = zip(*batch_with_eob)
        for d, eob in zip(data_, end_of_batch_):
            data.extend(d)
            end_of_batch.extend(eob)
        del data_, end_of_batch_

        data_source, durations = zip(*data)

        batch = []
        for idx, flag in zip(data_source, end_of_batch):
            batch.append(int(idx))
            if flag:
                yield batch
                batch = []

def customize_collate_fn(data_batch):
    specs, texts,inds,texts_ori = zip(*data_batch)

    src = pad_sequence(specs)
    tgt = pad_sequence(texts)
    tgt_ori = pad_sequence(texts_ori)

    src_padded_len = src.shape[0]
    tgt_padded_len = tgt.shape[0]
    tgt_ori_padded_len = tgt_ori.shape[0]

    src_len = list(map(len, specs))
    tgt_len = list(map(len, texts))
    tgt_ori_len = list(map(len, texts_ori))

    bsz = src.shape[1]

    inp_padding_mask = (torch.arange(src_padded_len).expand(bsz, src_padded_len).T >= torch.tensor(src_len).expand(src_padded_len, bsz))
    out_padding_mask = (torch.arange(tgt_padded_len).expand(bsz, tgt_padded_len).T >= torch.tensor(tgt_len).expand(tgt_padded_len, bsz))[1:]
    out_ori_padding_mask = (torch.arange(tgt_ori_padded_len).expand(bsz, tgt_ori_padded_len).T >= torch.tensor(tgt_ori_len).expand(tgt_ori_padded_len, bsz))[1:]
    return src, tgt[:-1], tgt[1:], inp_padding_mask, out_padding_mask,inds,tgt_ori[:-1],tgt_ori[1:],out_ori_padding_mask


def construct_dataloader(C,vocab_dict,vocab_list,augmentation_config):
    noise_manifest, noise_duration = read_manifest(
        mode='train',
        manifest_path=C['noise_manifest'],
        max_duration=C['max_duration'],
        min_duration=C['min_duration'], 
        max_text_len=C['max_text_len'],
    )

    noise_dataset = DatasetXL(
        manifest=noise_manifest,
        vocab_dict=vocab_dict,
        vocab_list=vocab_list,
        sample_rate=C['target_sample_rate'], 
        num_concat=C['concat_size'],
        n_fft=C['n_fft'], 
        win_len=C['win_len'], 
        hop_len=C['hop_len'], 
        n_mels=C['n_mels'],
        augmentation_config=augmentation_config
    )
    print(f"noise set size {len(noise_dataset)}")
    noise_dataloader = DataLoader(
        dataset=noise_dataset,
        batch_sampler=BatchSamplerXL(
            data_source=range(len(noise_manifest)),
            durations=noise_duration,
            mode='train',
            max_bsz=C['batch_size'],
            max_batch_duration=C['batch_size_in_s2'],
            shuffle=C['shuffle']),
        collate_fn=customize_collate_fn,
        num_workers=C['num_proc_data'],
        pin_memory=True,
    )


    meta_manifest, meta_duration = read_manifest(
        mode='train',
        manifest_path=C['meta_manifest'],
        max_duration=C['max_duration'],
        min_duration=C['min_duration'], 
        max_text_len=C['max_text_len'],
    )

    meta_dataset = DatasetXL(
        manifest=meta_manifest,
        vocab_dict=vocab_dict,
        vocab_list=vocab_list,
        sample_rate=C['target_sample_rate'], 
        num_concat=C['concat_size'],
        n_fft=C['n_fft'], 
        win_len=C['win_len'], 
        hop_len=C['hop_len'], 
        n_mels=C['n_mels'],
        augmentation_config=augmentation_config if C['use_aug'] else None
    )
    print(f"meta set size {len(meta_dataset)}")

    meta_dataloader = DataLoader(
        dataset=meta_dataset,
        batch_sampler=BatchSamplerXL(
            data_source=range(len(meta_dataset)),
            durations=meta_duration,
            mode='train',
            max_bsz=C['dev_batch_size'],
            max_batch_duration=C['dev_batch_size_in_s2'],
            shuffle=C['shuffle']),
        collate_fn=customize_collate_fn,
        num_workers=C['num_proc_data'],
        pin_memory=True
    )

    metadev_manifest, metadev_duration = read_manifest(
        mode='dev',
        manifest_path=C['metadev_manifest'],
        max_duration=C['max_duration'],
        min_duration=C['min_duration'], 
        max_text_len=C['max_text_len']
    )

    metadev_dataset = DatasetXL(
        manifest=metadev_manifest,
        vocab_dict=vocab_dict,
        vocab_list=vocab_list,
        sample_rate=C['target_sample_rate'], 
        num_concat=C['concat_size'],
        n_fft=C['n_fft'], 
        win_len=C['win_len'], 
        hop_len=C['hop_len'], 
        n_mels=C['n_mels']
    )
    print(f"metadev set size {len(metadev_dataset)}")

    metadev_dataloader = DataLoader(
        dataset=metadev_dataset,
        batch_sampler=BatchSamplerXL(
            data_source=range(len(metadev_dataset)),
            durations=metadev_duration,
            mode='dev',
            max_bsz=C['dev_batch_size'],
            max_batch_duration=C['dev_batch_size_in_s2'],
            shuffle=False),
        collate_fn=customize_collate_fn,
        num_workers=C['num_proc_data'],
        pin_memory=True
    )


    dev_manifest, dev_duration = read_manifest(
        mode='dev',
        manifest_path=C['dev_manifest'],
        max_duration=C['max_duration'],
        min_duration=C['min_duration'], 
        max_text_len=C['max_text_len']
    )

    dev_dataset = DatasetXL(
        manifest=dev_manifest,
        vocab_dict=vocab_dict,
        vocab_list=vocab_list,
        sample_rate=C['target_sample_rate'], 
        num_concat=C['concat_size'],
        n_fft=C['n_fft'], 
        win_len=C['win_len'], 
        hop_len=C['hop_len'], 
        n_mels=C['n_mels']
    )
    print(f"dev set size {len(dev_dataset)}")

    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_sampler=BatchSamplerXL(
            data_source=range(len(dev_dataset)),
            durations=dev_duration,
            mode='dev',
            max_bsz=C['dev_batch_size'],
            max_batch_duration=C['dev_batch_size_in_s2'],
            shuffle=False),
        collate_fn=customize_collate_fn,
        num_workers=C['num_proc_data'],
        pin_memory=True
    )

    return noise_dataloader,meta_dataloader,metadev_dataloader,dev_dataloader