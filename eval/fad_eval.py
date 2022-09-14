from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    use_pca=False, 
    use_activation=False,
    verbose=True
)

background_fnames = "dataset/train/violin/"
eval_fnames = "runs/exp_test/testrun/ref2/"
fad_score_ref = frechet.score(background_fnames, eval_fnames)

background_fnames = "dataset/train/violin/"
eval_fnames = "runs/exp_test/testrun/synth2/"
fad_score_synth = frechet.score(background_fnames, eval_fnames)

print("Original test set: ", fad_score_ref)
print("Synthesized test set: ", fad_score_synth)

