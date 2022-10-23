
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',mode='-'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode
        if self.mode == '-':
            self.signal_limit = np.Inf
        else:
            self.signal_limit = -np.Inf
        
    def __call__(self, signal, logger):
        if self.mode == '-':
            score = -signal
            if self.best_score is None:
                self.best_score = score
                
            elif score <= self.best_score + self.delta:
                self.counter += 1
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                if self.signal_limit > signal:
                    temp = self.signal_limit
                    self.signal_limit = signal
                    if self.verbose:
                        logger.info(f'Signal decreased ({temp:.6f} --> {signal:.6f}).  ')
                self.counter = 0
        else:
            score = signal
            if self.best_score is None:
                self.best_score = score
                
            elif score <= self.best_score + self.delta:
                self.counter += 1
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                if self.signal_limit < signal:
                    temp = self.signal_limit
                    self.signal_limit = signal
                    if self.verbose:
                        logger.info(f'Signal increased ({temp:.6f} --> {signal:.6f}).  ')
                self.counter = 0
