"""
TODO: not using hydra for now, but should convert back to hydra
"""

from ddx7.models import DDSP_Decoder, TCNFMDecoder
from ddx7.synth import FMSynth
from dataset.create_data import ProcessData 
from ddx7.data_utils.preprocessor import F0LoudnessRMSPreprocessor
from ddx7.loss_functions import rec_loss
import yaml
import torch
import librosa
import soundfile as sf


with open("recipes/model/tcnres_f0ld_fmstr.yaml", 'r') as f:
    config = yaml.safe_load(f)

with open("dataset/data_config.yaml", 'r') as f:
    data_config = yaml.safe_load(f)

# prepare model
decoder = TCNFMDecoder(n_blocks=config["decoder"]["n_blocks"], 
                        hidden_channels=config["decoder"]["hidden_channels"], 
                        out_channels=config["decoder"]["out_channels"],
                        kernel_size=config["decoder"]["kernel_size"],
                        dilation_base=config["decoder"]["dilation_base"],
                        apply_padding=config["decoder"]["apply_padding"],
                        deploy_residual=config["decoder"]["deploy_residual"],
                        input_keys=config["decoder"]["input_keys"])

synth = FMSynth(sample_rate=config["synth"]["sample_rate"],
                block_size=config["synth"]["block_size"],
                fr=config["synth"]["fr"],
                max_ol=config["synth"]["max_ol"],
                synth_module=config["synth"]["synth_module"])

model = DDSP_Decoder(decoder, synth)
model.load_state_dict(torch.load("runs/exp_test/testrun/state_best.pth"))

# prepare data
audio_file = "runs/exp_test/testrun/for_testing_only.wav"
audio, _ = librosa.load(audio_file, sr=16000)

preprocessor = ProcessData(
    silence_thresh_dB=data_config["data_processor"]["silence_thresh_dB"], 
    sr=data_config["data_processor"]["sr"], 
    device=data_config["data_processor"]["device"], 
    seq_len=data_config["data_processor"]["seq_len"],
    crepe_params=data_config["data_processor"]["crepe_params"], 
    loudness_params=data_config["data_processor"]["loudness_params"],
    rms_params=data_config["data_processor"]["rms_params"], 
    hop_size=data_config["data_processor"]["hop_size"], 
    max_len=data_config["data_processor"]["max_len"], 
    center=data_config["data_processor"]["center"]
)
f0 = preprocessor.extract_f0(audio)
loudness = preprocessor.calc_loudness(audio)
rms = preprocessor.calc_rms(audio)
x = {
    "audio": torch.tensor(audio).unsqueeze(0).unsqueeze(-1),
    "f0": torch.tensor(f0).unsqueeze(0).unsqueeze(-1),
    "loudness": torch.tensor(loudness).unsqueeze(0).unsqueeze(-1),
    "rms": torch.tensor(rms).unsqueeze(0).unsqueeze(-1)
}

scaler = F0LoudnessRMSPreprocessor()
scaler.run(x)

#print("INPUTS")
for k in x.keys():
   print(f'\t{k}: {x[k].size()} ')
print('')
# forward
synth_out = model(x)

print("OUTPUTS")
for k in synth_out.keys():
   print(f'\t{k}: {synth_out[k].size()} ')
print('')

loss = rec_loss(scales=[2048, 1024, 512, 256, 128, 64], overlap=0.75)(x['audio'],synth_out['synth_audio'])
print(loss)

sf.write("output.wav", synth_out["synth_audio"].squeeze().cpu().detach().numpy(), 16000)
