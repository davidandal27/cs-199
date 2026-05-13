import torch


def preprocess_audio(wav):
    wav = wav / (wav.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav



def randomized_smoothing(wav, sigma=0.001):
    if sigma <= 0:
        return wav

    noise = torch.randn_like(wav) * sigma
    wav = wav + noise
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav



def defend_audio(wav, sigma=0.001):
    wav = preprocess_audio(wav)
    wav = randomized_smoothing(wav, sigma=sigma)
    return wav