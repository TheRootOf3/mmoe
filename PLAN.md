TL;DR Trying to figure out whether it is possible to fuse outputs from the expert models just based on their confidence without the need for an external weighting model.


It is quite inconvenient that we have to use an external model that predicts the type of multimodal interaction of the input data. However, we don't quite need information about the interaction type (ternary classification in this context). It would be sufficient if each expert model applied one vs many binary classification, where the favoured interaction type is the type it specialises in. 

The question is whether we could extract this information purely from the estimates of model confidence...

What we have so far:
- Understanding of the original codebase and the provided resources. **Key takeaways**:
    - We have accessed to pre-processed multimodal datasets (including classification of unimodal inputs made by superior models -- hence, we also have access to interaction type classification.)
    - We have access to the logits from a single run of *ALL* expert models (yay! 🥳).
    - We have access to the logits from all mixing models (yay x2! 🥳).
    - We DO NOT have access to the models themselves!!! (not stonks 📉)
        - This is a serious limitation when it comes to replicating the experiments, because effectively we need to re-train all of the expert and mixing models. 
    - The original codebase had multiple issues -- it seems like the code has been changed in some places but not in others. The names of files, interaction types and paths are incorrect and need modifications. These have been mostly corrected by now! ✅
- Given there are no models available, it is tricky to reason about their confidence given a single record of logits from a single run.

Plan:
- [/] Need to train some models! Mixing models + small experts? 
    - [x] BLIP2 for each interaction type as a mixing model.
        - 2025-03-25 Done.
    - [/] Qwen2-0.5B as small experts -- 3 seeds: 32, 42, 2137.
        - 2025-03-25 Currently Training in progress.
- [x] We need to understand the confidence of each expert model {R,U,AS} on data from each interaction type {R,U,AS}.
    - 2025-03-25 Done! Added plotting in `./expert_fusion/fusion_analysis.py`.
- [ ] Furthermore, we need more information regarding methods for predicting confidence of classifier outputs.
    - [x] On Calibration of Modern Neural Networks, Guo et al., https://arxiv.org/abs/1706.04599
        - 2025-03-25 Good resource, worth studying in depth.
    - [ ] Temperature scaling in softmax?
        - Worth implementing, should be fairly straightforward with the current codebase.
    - [ ] EpiNets?
    - [ ] MC Dropout?
- [/] Need to measure the calibration of model confidence of the experts. 
    - [x] Should study the current confidence estimates. Can implement reliability diagrams, ECE.
        - 2025-03-26 Implemented rel diagrams and ECE per expert model. 
    - [ ] If we want to follow this direction then we need to implement the Softmax Temperature Scaling model fine-tuning.
- [ ] Write a summary of the datasets used.
- [ ] Do a bit more thorough literature review if we are following the direction of estimating confidence and model uncertainty. 
- [ ] Start writing a more clear version of this document and prepare a report template.





 

