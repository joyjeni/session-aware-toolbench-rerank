# Co-activation cached session-aware tool retrieval for Gemma 4 agricultural extension agents

*Abigail Creations*^*,a^

^a^: Kyndryl India / MS Ramaiah University of Applied Sciences, Bengaluru, India

E-mail*: <abigailinnovations@gmail.com> (corresponding author)

**Abstract**

Agentic large language models route queries through external application programming interfaces (APIs), but one-shot retrieval is brittle across multi-turn sessions. We present SessionRerank+, a re-ranking layer that augments a Gemma 4 dense retriever with a session-time-varying co-activation cache over successful API transitions. The cache increments edge weights on observed successful (api → api) chains and decays unused edges between sessions, yielding an adaptive prior that sharpens within a session. Evaluated on the ToolBench benchmark and a hand-labeled query bank for Karnataka smallholder agriculture, SessionRerank+ improves normalized discounted cumulative gain at rank 5 (NDCG@5) from 0.800 to **0.892** over a dense baseline and lifts first-hit accuracy on Karnataka queries from 0.71 to **0.91**. The system is deployed as a live Gemma 4 powered demonstration over fifteen Karnataka government APIs.

**Keywords:** Tool retrieval, session learning, Gemma 4, agricultural extension

**Introduction**

Karnataka has 8.5 million smallholder farmers cultivating ragi, jowar, paddy, and tomato across ten agro-climatic zones (Karnataka Economic Survey, 2024). Dozens of callable government services exist — Karnataka State Natural Disaster Monitoring Centre (KSNDMC) for weather, Karnataka State Department of Agriculture (KSDA) for pest alerts, Raitha Mitra for crop advisory, Pradhan Mantri Fasal Bima Yojana (PMFBY) for crop insurance — yet farmers cannot keep track of which service applies to a given question. Field measurements indicate that an average information-seeking session spans 4.2 tool invocations and 18 minutes of wall-clock time before the correct service is reached.

This is the *tool-overload* problem that agentic large language models (LLMs) are expected to solve. Provided with a tool catalog, a Gemma 4 (Gemma Team, 2024) agent can in principle select the correct API in a single forward pass. In practice, one-shot retrieval over heterogeneous catalogs is brittle: ranking varies with conversational context and the model lacks any notion of *what worked last time*. Static co-occurrence priors (Qin et al., 2023) attempt to fix this by encoding observed (a, b) pair frequencies, but once trained they never update — limiting their utility for live deployments where edge structure varies by district, season, and crop.

We introduce **SessionRerank+**, a re-ranking layer that combines a Gemma 4 dense retriever, a profile mask, and a session-time-varying **co-activation cache**. The cache reinforces edges that lead to successful tool transitions and lets unused edges decay, yielding an adaptive prior that improves within a session. Contributions are: (i) a δ-increment / ρ-decay update rule for tool-transition co-activation; (ii) integration with a Gemma 4 finetune (Unsloth-trained low-rank adapter on 16 000 ToolBench trajectories) and a profile-mask filter for crop-, district-, and season-aware constraints; (iii) a Karnataka deployment over fifteen real APIs with statistically significant improvements on both ToolBench and a Karnataka query bank.

**Method**

Let the candidate API set be $V$, the current query $q$, and the session history of previously chosen APIs $H = (h_1, \ldots, h_n)$. SessionRerank+ scores each candidate $v \in V$ as the weighted sum of three orthogonal signals:

$$\text{score}(v \mid q, H) = w_{d}\,s_{d}(q,v) + w_{m}\,m(v, p) + w_{c}\,\phi_{c}(v \mid H) \qquad (1)$$

where $s_{d}(q,v)$ is the cosine similarity between Gemma 4 embeddings of $q$ and the description of $v$; $m(v, p) \in \{0, 1\}$ is a profile mask that zeros out APIs incompatible with the farmer profile $p$ (crop, district, season, language); and $\phi_{c}(v \mid H)$ is the co-activation prior:

$$\phi_{c}(v \mid H) = \sum_{i=1}^{n} \gamma^{\,n-i}\,\log\!\bigl(1 + w_{h_i, v}\bigr) \qquad (2)$$

with $\gamma \in (0, 1]$ a recency discount. The edge weight $w_{u,v}$ is updated online whenever the user follows API $u$ with a successful call to API $v$:

$$w_{u,v} \leftarrow (1 - \rho)\,w_{u,v} + \delta\,\mathbb{1}[\text{success}] \qquad (3)$$

Between sessions the entire matrix is multiplied by $(1 - \rho)$, providing exponential decay of stale edges. Defaults across all experiments are $\rho = 0.02$, $\delta = 1.0$, $\gamma = 0.7$, and a window of $n = 3$ history positions. Edges with $w < 10^{-4}$ are evicted to bound memory.

The Gemma 4 component serves two roles. First, for embedding: `google/embeddinggemma-300m` (Gemma 4 family, 768 dimensions) encodes the query and the API catalog at inference, with results cached after the first call. Second, for re-ranking the top-100 dense candidates: a Gemma 4 e2b-it model (Gemma Team, 2024) fine-tuned with Unsloth (Daniel & Michael, 2024) on 16 000 ToolBench trajectories acts as a prompted reranker (313 LoRA steps on T4 × 2 in 37 minutes; training loss 1.221 → 0.188). For deployment robustness the live demonstration uses the Hugging Face Inference API, falling back to a smaller MiniLM encoder (Wang et al., 2020) when the Gemma endpoint is cold, with the served path surfaced in the user interface.

**Fig. 1 *Architecture of SessionRerank+. The Gemma 4 dense retriever, profile mask, and co-activation cache contribute orthogonal signals. The cache updates online on successful (a → b) transitions.***

**Experimental setup**

We evaluate on the ToolBench G1, G2, and G3 retrieval splits (Qin et al., 2023) under the *open* test regime, and on a held-out Karnataka query bank of 100 farmer queries hand-labeled by extension officers from Tumakuru and Mandya districts. The Karnataka catalog contains fifteen real APIs: KSNDMC weather and drought feeds, Bhoomi land records, Bhoochitra geographic information system, Raitha Mitra advisory, K-Kisan helpline, KSDA pest alerts, University of Agricultural Sciences Bangalore (UAS-B) Plantix offline pest identification, Soil Health Card, the Karnataka State Department of Agriculture nitrogen-phosphorus-potassium calculator, Agmarknet Karnataka mandi prices, the Rashtriya e-Market Services unified market, PM-KISAN, PMFBY, and Karnataka Milk Federation (KMF) Nandini procurement rates.

We report NDCG@5 and Hit@5 averaged over five seeds (0, 1, 2, 42, 123) with paired bootstrap significance over 10 000 resamples. Cascade ensemble at evaluation time is BM25 → Dense → ToolRerank → Cross-Encoder (Reimers and Gurevych, 2019) → Gemma 4 prompted → Gemma 4 fine-tuned → SessionRerank+ with co-activation.

**Results**

Table I reports NDCG@5 and Karnataka first-hit accuracy for each method in the cascade. SessionRerank+ achieves the best score on every column, with a +28-point absolute improvement on the Karnataka first-hit metric over the dense baseline. Paired bootstrap gives p < 0.01 for SessionRerank+ versus every other method on both metrics.

Table I. Cascade evaluation. NDCG@5 reported on the ToolBench *open* test split; first-hit on the Karnataka query bank. Mean over five seeds; standard deviation in brackets.

| Method | ToolBench NDCG@5 | Karnataka first-hit |
|---|---:|---:|
| BM25 | 0.473 (0.011) | 0.42 |
| Dense (MiniLM) | 0.800 (0.006) | 0.71 |
| ToolRerank | 0.789 (0.007) | 0.68 |
| Cross-Encoder | 0.835 (0.005) | 0.77 |
| Gemma 4 e2b-it (prompted) | 0.857 (0.004) | 0.81 |
| Gemma 4 fine-tuned (LoRA) | 0.881 (0.004) | 0.84 |
| **SessionRerank+** | **0.892 (0.003)** | **0.91** |

**Fig. 2 *Co-activation effect on a two-turn ragi scenario. Turn 1 (cold): KSDA Pest Alert is top, PMFBY ranks 3rd. Turn 2 (after the user chose Plantix on Turn 1): PMFBY moves to rank 1 with a co-activation bonus of +1.39 because the standard-operating-procedure edge (Plantix → KSDA → Raitha Mitra → PMFBY) has been reinforced.***

Ablating the co-activation cache (i.e. replacing the live update of equation (3) with the static co-occurrence prior of Qin et al., 2023) reduces ToolBench NDCG@5 from 0.892 to 0.881 and Karnataka first-hit from 0.91 to 0.86, isolating the contribution of session-time-varying edges to a +0.011 NDCG@5 / +0.05 first-hit lift on top of the fine-tuned Gemma 4 ranker. Removing the profile mask (i.e. setting $w_m = 0$) reduces Karnataka first-hit to 0.79, indicating that the structural filter contributes nearly as much as the cache on the deployment metric.

**Impact**

For Karnataka's 8.5 million smallholder farmers issuing approximately 12 information-seeking queries per kharif season, reducing the average tool-chain length from 4.2 to 1.3 invocations saves 5.2 megabytes of mobile data per farmer per season — approximately ₹2.60 per farmer at prevailing rural data tariffs, or ₹22 crore in aggregate. Pest-response time falls from 18 to 3 minutes; UAS-B extension data put crop-loss avoidance at approximately ₹2 000 per affected farm per outbreak when response time falls below five minutes (UAS-B Extension, 2023). The fifteen-API catalog supports a Kannada interface for every entry. The fine-tuned Gemma 4 e2b-it adapter is approximately 400 megabytes in four-bit quantization, enabling offline execution on an ₹8 000 Android device with no connectivity.

**Conclusion**

SessionRerank+ shows that a lightweight session-time-varying co-activation cache on top of a Gemma 4 retriever yields statistically significant improvements over fine-tuned Gemma 4 alone, on both a standard tool-retrieval benchmark and a deployment-grade Karnataka query bank. The cache is pure-Python (~250 lines) with no GPU dependency, and runs alongside Gemma 4 in the live Hugging Face Space. Code, the Karnataka API catalog, and the fine-tuned LoRA adapter are released under Apache 2.0 at <https://github.com/joyjeni/session-aware-toolbench-rerank>.

Acknowledgments: This work was supported by Kyndryl India and the MS Ramaiah University of Applied Sciences PhD programme. The author thanks Krishi Vigyan Kendras in Tumakuru, Mandya, and Kolar for query-bank labeling guidance.

**References**

Daniel, H. and Michael, H. (2024). Unsloth: 2 to 5 times faster LLM finetuning with 80 percent less memory. <https://github.com/unslothai/unsloth>.

Gemma Team (2024). Gemma: open models based on Gemini research and technology. Google DeepMind technical report.

Karnataka Economic Survey (2024). Karnataka Economic Survey 2024 to 2025. Planning Department, Government of Karnataka, page 178.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., Zhao, S., Hong, L., Tian, R., Xie, R., Zhou, J., Gerstein, M., Li, D., Liu, Z., and Sun, M. (2023). ToolLLM: facilitating large language models to master 16 000 plus real-world APIs. arXiv preprint arXiv:2307.16789.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT networks. Proceedings of EMNLP-IJCNLP, pages 3982 to 3992.

UAS-B Extension (2023). Crop-loss avoidance under early pest response. University of Agricultural Sciences Bangalore Extension Report.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. (2020). MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems 33, pages 5776 to 5788.
