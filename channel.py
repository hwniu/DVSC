import numpy as np
import torch
import math

class Channel(object):
    """"implement power constraint"""

    def __init__(self, channel_type, name='channel'):
        self.channel_type = channel_type
        self.name = name

    def __call__(self, semantic_feature, snr_db=None, rician_factor=None):
        inter_shape = [item for item in semantic_feature.shape]
        f_temp = semantic_feature.view(semantic_feature.shape[0], -1)
        dim_z = f_temp.shape[1] // 2
        f_channel_in = torch.complex(f_temp[:, :dim_z], f_temp[:, dim_z:])
        # 归一化能量
        norm_factor = torch.sum(torch.real(f_channel_in * torch.conj(f_channel_in)), dim=1, keepdim=True)
        f_channel_norm_in = f_channel_in * torch.complex(torch.sqrt(float(dim_z)/norm_factor), torch.Tensor([0.0]).cuda())
        if self.channel_type == 'awgn':
            if snr_db is None:
                raise Exception("This input snr should exist!")
            semantic_feature_out = awgn(f_channel_norm_in, snr_db)
        elif self.channel_type == 'rayleigh':
            if snr_db is None:
                raise Exception("This input snr should exist!")
            semantic_feature_out = rayleigh_fading_channel(f_channel_norm_in, snr_db)
        elif self.channel_type == 'rician':
            if snr_db is None or rician_factor is None:
                raise Exception("This input snr should exist and the rician_factor should exist!")
            semantic_feature_out = rician_fading_channel(f_channel_norm_in, snr_db, rician_factor)
        elif self.channel_type == 'fisher_snedecor':
            if snr_db is None:
                raise Exception("This input snr should exist!")
            semantic_feature_out = fisher_snedecor_f_fading_channel(f_channel_norm_in, snr_db)
        else:
            raise Exception("This option shouldn't be an option!")

        semantic_feature_out = torch.concat([torch.real(semantic_feature_out), torch.imag(semantic_feature_out)], 1)
        semantic_feature_out = semantic_feature_out.reshape(inter_shape)
        return semantic_feature_out

def awgn(x, snr_db):
    noise_stddev = math.sqrt(10 ** (-snr_db / 10))
    noise_stddev = complex(noise_stddev, 0.)
    awgn_noise_channel = torch.complex(torch.normal(0, 1 / np.sqrt(2), x.shape), torch.normal(0, 1 / np.sqrt(2), x.shape)).cuda()
    return x + noise_stddev * awgn_noise_channel

def rician_fading_channel(x, snr_db, rician_factor):
    ### E||x||^2 = 1 ##归一化功率限制
    ### Rician Channel ###
    noise_stddev = math.sqrt(10 ** (-snr_db / 10))
    noise_stddev = complex(noise_stddev, 0.)
    awgn_noise_channel = torch.complex(torch.normal(0, 1 / np.sqrt(2), x.shape), torch.normal(0, 1 / np.sqrt(2), x.shape)).cuda()
    # channel_gain
    h_real = torch.normal(np.sqrt(rician_factor/(1 + rician_factor)), np.sqrt(1/(2*(rician_factor + 1))), x.shape)
    h_imag = torch.normal(np.sqrt(rician_factor/(1 + rician_factor)), np.sqrt(1/(2*(rician_factor + 1))), x.shape)
    h = torch.complex(real=h_real, imag=h_imag).cuda()
    return x * h + noise_stddev * awgn_noise_channel

def rayleigh_fading_channel(x, snr_db):
    ### Rayleigh Channel ###
    noise_stddev = math.sqrt(10 ** (-snr_db / 10))
    noise_stddev = complex(noise_stddev, 0.)
    awgn_noise_channel = torch.complex(torch.normal(0, 1 / np.sqrt(2), x.shape), torch.normal(0, 1 / np.sqrt(2), x.shape)).cuda()
    # channel gain
    h_real = torch.normal(0, 1 / np.sqrt(2), x.shape)
    h_imag = torch.normal(0, 1 / np.sqrt(2), x.shape)
    h = torch.complex(real=h_real, imag=h_imag).cuda()
    return x * h + noise_stddev * awgn_noise_channel


def fisher_snedecor_f_fading_channel(iq_signal, snr_db):
    """
    Implements the Fisher-Snedecor F fading complex channel with input IQ signal and input signal power of 1 and signal-to-noise ratio in dB.

    :param iq_signal: The input IQ signal to the channel.
    :param snr_db: The signal-to-noise ratio in dB.
    :return: The output IQ signal after passing through the channel.
    """
    # Convert SNR from dB to linear scale
    snr = 10 ** (snr_db / 10)

    # Generate the fading coefficients with Fisher-Snedecor F distribution
    m = 2
    n = 2
    fading_coeff_1 = np.random.f(m, n)
    fading_coeff_2 = np.random.f(m, n)
    fading_coeff = fading_coeff_1 + 1j * fading_coeff_2

    # Generate the noise with zero mean and unit variance
    noise_stddev = math.sqrt(10 ** (-snr_db / 10))
    noise_stddev = complex(noise_stddev, 0.)
    awgn_noise_channel = torch.complex(torch.normal(0, 1 / np.sqrt(2), iq_signal.shape), torch.normal(0, 1 / np.sqrt(2), iq_signal.shape)).cuda()

    # Compute the output IQ signal after passing through the channel
    output_iq_signal = np.sqrt(snr / fading_coeff) * iq_signal + noise_stddev * awgn_noise_channel

    return output_iq_signal
