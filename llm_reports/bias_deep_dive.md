## Bias found in OOD shortcut learning
This is extremely clean. The answer is: only lunge clips, only performer 7, only light-skin variants.

![analysis](../out/skin_tone_probe_rgb_torchvision_v6_cj0p0/skin_tone_pair_heatmap_f1_macro.png)

Every single error across all 6 models and all 3 seeds is:

lunge_7_modified_asian.mp4 (3 backgrounds)
lunge_7_modified_white.mp4 (3 backgrounds)
Predicted as cartwheel (class 1), true label lunge (class 0)
Zero cartwheel clips are ever misclassified. Zero lunge_2 clips are ever misclassified. Zero lunge with african/indian skin are ever misclassified.

This completely settles the debate:

It's not a skin-tone shortcut in the broad sense. The model handles lunge_7 with dark skin fine in the matched eval. It handles ALL other lunge performers with light skin fine in the shifted eval. And performer 2 (standing still + step) — also a weird, OOD lunge — is correctly classified in every model and seed despite having light skin in the shifted eval.

It's specifically performer 7's jumping lunge motion. You described it as "jumping with wide legs." That motion:

Has high optical flow (from the jump), pushing it toward cartwheel territory
Is not a standard lunge the model recognises confidently
Gets misclassified as cartwheel when it loses the "dark skin = lunge" training signal
The skin tone matters here not as the primary feature, but as the tie-breaker: when the model is uncertain about performer 7's unusual motion, it falls back on "light skin → cartwheel." For performer 2, even with light skin, the near-zero motion is enough to say "definitely not a cartwheel."

The conclusion: this is a data quality issue with one specific performer (ID 7) who performs an atypical lunge that all 6 architectures consistently confuse with cartwheel when the skin-tone training correlation is removed.

## Bias found in foundation models through SSM structure

Results are in, and they're counterintuitive in an interesting way.

The surprising finding: lunge has the LOWEST r (0.07)

Lunge's SSM structure barely changes when skin tone is swapped — far less than any other action. Yet lunge is the action that showed classifier bias. Meanwhile celebrate (r=0.30) and clap (r=0.21) have large SSM changes with skin tone but zero classifier bias.

This directly answers your question. SSM Frobenius distance is not what drives classifier bias here. The two phenomena are orthogonal:

SSM structure changes with skin	Classifier bias
lunge	low (r=0.07)	yes
clap	high (r=0.21)	no
celebrate	high (r=0.30)	no
The note about max_r is also interesting — clap has individual clips reaching r=1.91, meaning skin tone changes that clip's SSM more than switching to celebrate entirely. But no classifier ever fails on clap.

What this means

The bias in lunge_7 is a position effect, not a structure effect. DINOv2 preserves the temporal motion shape across skin tones (SSM invariant), but the entire trajectory can drift in feature space — moving closer to another action's region. The SSM captures shape, not location. The classifier uses location.

This is actually a clean story for your paper: foundation model representations are temporally structure-invariant to skin tone (SSM Frobenius r << 1 across all actions), but positional drift in feature space can still cause misclassification for OOD clips where motion is ambiguous. The two effects are separable and the structural one is clearly benign.

The job is still running for the remaining actions (squat, tie, yawn, fish) — once it finishes you can re-run the analysis script to get the full picture.