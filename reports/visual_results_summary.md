# Visual Results Summary

This visual analysis phase is exploratory and should be read together with the final 50-epoch comparison in [ds007561_50ep_model_comparison.md](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/ds007561_50ep_model_comparison.md). The `ds007561` cohort is small and strongly imbalanced, with only one `AD` subject, so the figures below are research inspection outputs for representation analysis and model debugging, not clinical evidence.

## Latent Space

The final latent-space figures summarize subject embeddings extracted from the `50`-epoch multimodal, MRI-only, and PET-only checkpoints.

- `PCA` provides a linear global view of the learned representation and reports explained variance on each axis.
- `t-SNE` provides a local neighborhood view that is useful for checking whether subjects with similar labels tend to occupy nearby regions of the embedding space.
- Because of the small cohort and class imbalance, apparent separation or overlap should be treated as exploratory only.

Reference files:
- [multimodal_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep/multimodal_pca.png)
- [multimodal_pca.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep/multimodal_pca.svg)
- [multimodal_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep/multimodal_tsne.png)
- [multimodal_tsne.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep/multimodal_tsne.svg)
- [mri_only_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_mri_only/mri_only_pca.png)
- [mri_only_pca.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_mri_only/mri_only_pca.svg)
- [mri_only_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_mri_only/mri_only_tsne.png)
- [mri_only_tsne.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_mri_only/mri_only_tsne.svg)
- [pet_only_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_pet_only/pet_only_pca.png)
- [pet_only_pca.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_pet_only/pet_only_pca.svg)
- [pet_only_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_pet_only/pet_only_tsne.png)
- [pet_only_tsne.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep_pet_only/pet_only_tsne.svg)

## Refined Grad-CAM

The final multimodal Grad-CAM overlays were generated from the `50`-epoch multimodal checkpoint and refined to produce cleaner and more anatomically plausible activation maps.

Refinements applied:
- MRI-derived soft brain masking to suppress activations outside the brain
- percentile thresholding to retain only the strongest activations
- Gaussian smoothing to reduce block-like artifacts and improve spatial continuity
- consistent slice selection, centered cropping, and aligned MRI/PET visualization

These overlays are therefore more suitable for exploratory model inspection than the earlier raw outputs. Even so, they remain model-derived activation summaries, not anatomical ground truth and not clinical biomarkers.

Reference files:
- [sub-09_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-09_multimodal_gradcam.png)
- [sub-09_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-09_multimodal_gradcam.svg)
- [sub-13_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-13_multimodal_gradcam.png)
- [sub-13_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-13_multimodal_gradcam.svg)
- [sub-16_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-16_multimodal_gradcam.png)
- [sub-16_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-16_multimodal_gradcam.svg)
- [figure_selection_manifest.json](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/figure_selection_manifest.json)

## Atlas Labeling Status

Atlas-based anatomical labeling was intentionally skipped. No registered atlas is currently available in the project, and subject volumes have not been mapped into a shared atlas space, so assigning anatomical region names at this stage would be unreliable.

For that reason, the activation maps should be treated as exploratory visual evidence of model focus only. They are not anatomical proof and should not be interpreted as clinical localization findings.

Reference:
- [atlas_region_summary.json](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/atlas_region_summary.json)
