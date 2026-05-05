# Visual Results Summary

This visual analysis phase is exploratory only. The `ds007561` cohort used here is very small and strongly imbalanced, with only one `AD` subject, so these figures should be treated as technical inspection outputs rather than evidence of clinical performance.

## Latent Embeddings

The latent-space figures show how subject-level embeddings from the trained smoke-test checkpoints are arranged after dimensionality reduction. `PCA` gives a linear global view of the embedding structure, while `t-SNE` gives a local neighborhood view that can highlight small clusters or overlaps between `Control`, `MCI`, and `AD`.

Multimodal:
- [multimodal_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/multimodal_pca.png)
- [multimodal_pca.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/multimodal_pca.svg)
- [multimodal_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/multimodal_tsne.png)
- [multimodal_tsne.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/multimodal_tsne.svg)

MRI-only:
- [mri_only_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/mri_only_pca.png)
- [mri_only_pca.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/mri_only_pca.svg)
- [mri_only_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/mri_only_tsne.png)
- [mri_only_tsne.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/mri_only_tsne.svg)

PET-only:
- [pet_only_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/pet_only_pca.png)
- [pet_only_pca.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/pet_only_pca.svg)
- [pet_only_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/pet_only_tsne.png)
- [pet_only_tsne.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/pet_only_tsne.svg)

## Grad-CAM Overlays

The Grad-CAM figures show slice-level activation overlays for a few selected multimodal subjects. These overlays highlight image regions that contributed more strongly to the model output for the predicted class. They are useful for debugging whether the model is focusing on structured brain content rather than obvious artifacts, but they should not be interpreted as anatomical biomarkers.

Generated exploratory overlays:
- [sub-08_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-08_multimodal_gradcam.png)
- [sub-08_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-08_multimodal_gradcam.svg)
- [sub-11_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-11_multimodal_gradcam.png)
- [sub-11_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-11_multimodal_gradcam.svg)
- [sub-15_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-15_multimodal_gradcam.png)
- [sub-15_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-15_multimodal_gradcam.svg)

## Atlas Status

Atlas-guided anatomical labeling was intentionally skipped at this stage. The current pipeline does not yet include subject-to-atlas registration, and no registered atlas files were provided locally, so assigning anatomical region names now would be unreliable.

Reference:
- [atlas_region_summary.json](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/atlas_region_summary.json)

## Scope Note

These figures are useful for checking representation structure, sanity of model focus, and whether multimodal fusion appears technically different from unimodal baselines. They are not sufficient to support medical or clinical claims.
