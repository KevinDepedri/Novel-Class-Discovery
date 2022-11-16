# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np
# current is the training step or the epoch
# ramp length i donot get what is it??? 
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        # For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, 
        # and values larger than 1 become 1.
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    epoch= np.arange(1,200)
    data = [sigmoid_rampup(i,50) for i in epoch]
    plt.plot(epoch,data,'g',label='sigmoid_rampup') 
    data = [linear_rampup(i,50) for i in epoch]
    plt.plot(epoch,data,'red',label='linear_rampup') 
    data = [cosine_rampdown(i,200) for i in epoch]
    plt.plot(epoch,data,'b',label='cosine_rampdown')
    plt.xlabel('epoch')
    plt.ylabel('values')
    plt.legend()

    plt.show()