import os
import numpy as np
from pkg.utils import get_spectrum, spectrogram2wav, plot_attention, PrettyBar, guide_attention
from pkg.hyper import Hyper
from pkg.data import load_data
import scipy.io.wavfile as wavfile


def process_file(path, text_len):
    # e.g.
    # "~/Desktop/python/03.wav" 일 때 fname=03 / ext=wav
    fname, ext = os.path.splitext(os.path.basename(path))
    if ext != ".wav":
        raise Exception("[preprocess]: only support wav file")

    ## a = ([[1,2,3],[4,5,6]]) -> a.shape == (2,3)
    mel, mag = get_spectrum(path)
    t = mel.shape[1]

    # right padding, reduce shape
    pad = Hyper.temporal_rate - (t % Hyper.temporal_rate) if t % Hyper.temporal_rate != 0 else 0
    mel = np.pad(mel, [[0, 0], [0, pad]], mode="constant")
    mag = np.pad(mag, [[0, 0], [0, pad]], mode="constant")

    # temporal reduction
    mel = mel[..., ::Hyper.temporal_rate]

    mel_path = os.path.join(Hyper.feat_dir, "mels")
    mag_path = os.path.join(Hyper.feat_dir, "mags")
    if not os.path.exists(mel_path):
        os.makedirs(mel_path)
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)
    np.save(os.path.join(mel_path, fname + ".npy"), mel.astype(np.float32))
    np.save(os.path.join(mag_path, fname + ".npy"), mag.astype(np.float32))

    # # attention guide
    guide_path = os.path.join(Hyper.feat_dir, "guides")
    mask_path = os.path.join(Hyper.feat_dir, "masks")
    if not os.path.exists(guide_path):
        os.makedirs(guide_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    ## mel.shape[-1] -> 열 개수
    ## text_len -> .wav 파일의 텍스트를 숫자로 표현한 배열의 크기 ex)[2,3]
    ##[text_len] -> [ [2,3] ]
    guide, mask = guide_attention([text_len], [mel.shape[-1]],
                                  Hyper.data_max_text_length,
                                  Hyper.data_max_mel_length)

    ## 2차 배열 guide 에서 첫번째 행만 넘김
    guide = guide[0]
    mask = mask[0]
    np.save(os.path.join(guide_path, fname + ".npy"), guide.astype(np.float32))
    np.save(os.path.join(mask_path, fname + ".npy"), mask.astype(np.float32))
    # plot_attention(guide, "guide", 0, True)
    # plot_attention(mask, "mask", 0, True)


def preprocess():
    print("pre-processing...")

    ## names, lengths,texts -> array 
    ## lengths: len of txt 문자열 길이
    ## names: ['LJ-001-0002', 'LJ001-0008'] 파일 이름 배열로
    ## texts: [ [11,3,5,2,...8], [4,5,6,7...,2] ] 문장 숫자로 변환한 것들 배열로
    names, lengths, texts = load_data()
    bar = PrettyBar(len(names))
    for i in bar:
        fname = names[i]
        fpath = os.path.join(Hyper.data_dir, "wavs/" + fname + ".wav")
        bar.set_description(fname)
        process_file(fpath, lengths[i])


if __name__ == "__main__":
    preprocess()
