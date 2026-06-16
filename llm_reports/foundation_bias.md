I have enough to give you a grounded conceptual answer. Short version: yes, this is possible, your intuition is well-founded, and the specific angle you're describing — temporal-dynamics fairness rather than static-embedding fairness — is genuinely under-explored. The tools to do it well already exist, but they come from view/appearance-invariance research, not fairness research, which is exactly why the intersection is interesting.

1. The conceptual core: decompose the trajectory into "offset" vs "shape"
Think of a video as a path through embedding space: a sequence $e(1), e(2), \dots, e(T)$, one vector per frame. For the same avatar doing the same motion with only skin tone changed, you can decompose the dark-vs-light difference into two parts:

Static offset — the difference of the per-video means, $|\bar e_{\text{dark}} - \bar e_{\text{light}}|$. This is the "skin and light reflect differently" part. You expect this to be non-zero, and it's not really bias in the harmful sense — it's the model correctly seeing a different-colored surface.
Dynamic / shape — the difference between the mean-centered trajectories $\tilde e(t) = e(t) - \bar e$. This is how the embedding moves as the action unfolds.
Your hypothesis, stated precisely: a fair model puts almost all of the dark–light difference into the static offset, and leaves the shape near-identical. Bias = a systematic shape difference, i.e., skin tone leaking into the motion/dynamics subspace rather than staying in the appearance subspace. This is exactly the "disentangle appearance from motion" framing, which is an active research area in its own right (G3AN, DisMo, MoAlign) — those papers confirm the decomposition is principled, and they explicitly note motion features tend to get entangled with "object textures, lighting conditions, and background."

2. The key tool: temporal self-similarity matrices (appearance-invariant by construction)
The cleanest way to measure "shape" while throwing away the static offset is a temporal self-similarity matrix (SSM): $M[i,j] = \text{sim}(e(i), e(j))$ over all frame pairs. The crucial property, established back in Junejo et al., ECCV 2008 and used as the central representation in RepNet (Dwibedi et al., CVPR 2020): the SSM is approximately invariant to viewpoint and appearance — it depends only on the relationships between frames, not their absolute position. If poses at $t_1$ and $t_2$ are similar, $M$ is low for any rendering of that action.

That's almost custom-built for your question. Build the SSM from frozen CLIP/DINO per-frame embeddings for the dark video and the light video, then compare them (e.g. Frobenius distance $|M_{\text{dark}} - M_{\text{light}}|_F$). A fair model → near-identical SSMs even though the raw embeddings sit in different regions. A divergence in the SSMs is direct evidence that skin tone is bending the temporal structure of the action.

3. Your synthetic dataset is a major asset here
Most skin-tone fairness work cannot isolate skin tone — in real footage, skin tone is confounded with identity, clothing, lighting, location. Meta's PUG (NeurIPS 2023) made exactly this argument and built an Unreal-Engine dataset to vary one factor at a time for fairness/robustness probing. Your benchmark is the same paradigm — and better for this purpose, because you likely have frame-aligned pairs: same motion, same frame index ⇒ same action phase across skin tones. That gives you two things for free:

You can compare $e_{\text{dark}}(t)$ vs $e_{\text{light}}(t)$ directly, frame by frame, no temporal alignment needed.
You have a built-in null model: vary identity/seed within the same skin tone and same action. That tells you how much shape variation is "normal," so you can ask whether across-skin-tone shape difference exceeds within-skin-tone variation. Without that control you can't tell a real bias from noise.
If you later move to non-aligned data, the alignment problem itself has a clean tool: Temporal Cycle-Consistency (TCC), Dwibedi et al. CVPR 2019 learns per-frame embeddings that align action phases across videos. You could even repurpose it as a probe: does a dark video align to its light counterpart in phase as cleanly as two same-skin-tone videos do? Asymmetry in alignment quality = bias.

4. The honest caveat you need to design around
Skin tone is not a pure additive offset. Darker and lighter surfaces interact with lighting non-linearly (specular highlights, shadow contrast, dynamic range), and as limbs move and self-occlude, the amount of visible skin changes over the action. So even a perfectly fair model could show some real, physically-grounded skin-tone-dependent trajectory variation — that's the renderer, not the model's prejudice. This is why the within-skin-tone null model in §3 matters: you're not asking "is the shape difference zero" (it won't be), you're asking "is it larger than the variation you'd see from any other nuisance factor." Frame your headline metric as a ratio or fraction: of the total dark–light embedding gap, what share is static offset vs dynamics, and does the dynamics share exceed the within-group baseline.

5. Concrete shape-comparison metrics (menu, for when we get to code)
Given frame-aligned trajectories, in rough order of how appearance-invariant they are:

Mean-centered per-frame cosine/L2 — simplest; isolates dynamics after removing offset.
Velocity vectors $\Delta e(t) = e(t{+}1) - e(t)$ — compare direction and magnitude over time; the offset cancels automatically in the difference.
SSM Frobenius distance — §2, the strongest appearance-invariant choice.
Linear CKA / Procrustes residual — compare the geometry of the two trajectory point-clouds up to rotation/scale; CKA is the standard representational-similarity tool (Kornblith 2019).
The "can you predict skin tone from dynamics" probe — train a classifier on mean-centered velocity sequences to predict skin tone. If it succeeds, the dynamics provably encode skin tone. This is the temporal analog of the iEAT embedding-association tests used in static skin-tone bias work, and it ties directly back into your existing shortcut-probe framework.
6. Which models, and a caveat about "temporal"
Be aware: CLIP, DINOv2, EVA-02, SigLIP are per-frame image encoders with no temporal modeling. Your "embedding over frames" is a sequence of independent image embeddings; the trajectory is induced purely by the changing frame content. That's perfectly valid — arguably cleaner — but it means you're probing the image encoder's skin-tone sensitivity as pose evolves, not a learned temporal module. If you want models that natively model dynamics, use video foundation models: VideoMAE / VideoMAEv2, V-JEPA / V-JEPA2, InternVideo2, ViCLIP, Hiera, TimeSformer. For those you'd build the trajectory from per-clip sliding-window embeddings or patch/token sequences. Running both families is itself a finding: does adding temporal modeling reduce or amplify the dynamics-level skin-tone leakage?

7. Where this sits relative to prior work
Static foundation-model skin-tone bias is well-covered: embedding-association tests, Fairness of Discriminative Foundation Models in CV, CLIP skin-lightening biases, the Survey of Social Bias in VLMs. All of these ask "does the static embedding encode/associate skin tone." None ask your question.
Temporal bias in video is mostly studied in generation: e.g. "Temporal Attribute Stability" metrics that measure whether a generated person's attributes stay consistent across frames (video diffusion bias work), and video-FM trustworthiness benchmarks (VMDT). Conceptually adjacent, but about generative consistency, not representational fairness of dynamics.
The appearance-invariance tools (SSM, TCC, motion/appearance disentanglement) exist but were built for view-invariant recognition and motion transfer, not fairness.
So the gap you'd be filling: applying appearance-invariant temporal-representation tools to ask a fairness question — "is an action's representational trajectory invariant to skin tone" — on a controlled synthetic dataset that can actually isolate the variable. I couldn't find anything that does exactly that, which is a good sign for novelty and a clean story.

A couple of things worth deciding before any code, since they shape the whole evaluation:

Are your dark/light pairs truly frame-aligned (identical motion, identical frame timing)? If yes, the design simplifies enormously (direct frame-wise comparison, no DTW/TCC). Worth confirming against the actual rendered data.
What's the headline claim you want — a single "dynamics fairness" scalar per (model, action), or a richer per-frame/per-phase profile of where in the action the bias appears? The second is more interesting (e.g. "bias spikes during occlusion-heavy phases") but needs the aligned pairs.
Want me to sketch the evaluation protocol (which embeddings, which metrics, which controls, what the output tables/plots look like) before we touch implementation?

Sources:

Temporal Self-Similarity for view-invariant action recognition (Junejo et al., ECCV 2008)
RepNet / Counting Out Time (Dwibedi et al., CVPR 2020)
Temporal Cycle-Consistency Learning (Dwibedi et al., CVPR 2019)
PUG: Photorealistic & Semantically Controllable Synthetic Data (Meta, NeurIPS 2023)
Disentangling Appearance and Motion — G3AN, DisMo, MoAlign
Evaluating the Fairness of Discriminative Foundation Models in CV
Survey of Social Bias in Vision-Language Models
Role of Alignment Tuning in Social Bias in Video Diffusion Models (Temporal Attribute Stability)
VMDT: Trustworthiness of Video Foundation Models
Similarity of Neural Network Representations Revisited (CKA, Kornblith 2019)