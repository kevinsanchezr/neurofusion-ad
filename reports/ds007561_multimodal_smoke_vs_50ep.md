# ds007561 Multimodal Smoke vs 50-Epoch Comparison

This comparison is exploratory only. The cohort is very small and strongly imbalanced, with only one AD subject.

## Metrics Summary

| Run | Epochs | Best Val Epoch | Final Train Loss | Final Val Loss | Train Balanced Acc. | Val Balanced Acc. | Test Balanced Acc. | Train Macro F1 | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke test | 1 | 1 | 0.9814 | 0.9478 | 0.3444 | 0.5000 | 0.5000 | 0.3286 | 0.4000 | 0.4000 |
| 50-epoch exploratory run | 50 | 42 | 0.0138 | 0.6799 | 1.0000 | 0.5000 | 0.5000 | 1.0000 | 0.4000 | 0.4000 |

## Interpretation

- The 50-epoch run reduced training loss by `-0.9676` relative to smoke, while validation loss changed by `-0.2680`.
- Training balanced accuracy changed by `0.6556`, but validation balanced accuracy changed by `0.0000`.
- Training macro F1 changed by `0.6714`, while validation macro F1 changed by `0.0000`.
- Test balanced accuracy changed by `0.0000` and test macro F1 changed by `0.0000`.
- If training improves while validation and test remain flat, that should be read as overfitting rather than as a meaningful gain.

## Confusion Matrices

- Smoke train: `[[7, 3, 0], [2, 1, 0], [1, 0, 0]]`
- Smoke validation: `[[2, 0, 0], [1, 0, 0], [0, 0, 0]]`
- Smoke test: `[[2, 0, 0], [1, 0, 0], [0, 0, 0]]`
- 50-epoch train: `[[10, 0, 0], [0, 3, 0], [0, 0, 1]]`
- 50-epoch validation: `[[2, 0, 0], [1, 0, 0], [0, 0, 0]]`
- 50-epoch test: `[[2, 0, 0], [1, 0, 0], [0, 0, 0]]`

## Latent Space

- Smoke embeddings: `reports/experiments/embeddings/ds007561_multimodal_embeddings.npz`
- 50-epoch embeddings: `reports/experiments/embeddings_50ep/multimodal_50ep_multimodal_embeddings.npz`
- Smoke PCA variance ratio: `0.4040, 0.1095`
- 50-epoch PCA variance ratio: `0.6224, 0.1935`
- Smoke centroid distances: `{'Control_vs_MCI': 0.014243351566126772, 'Control_vs_AD': 0.03582096805682133, 'MCI_vs_AD': 0.04888649746499901}`
- 50-epoch centroid distances: `{'Control_vs_MCI': 2.1941608955603815, 'Control_vs_AD': 6.058555278634691, 'MCI_vs_AD': 4.531448619382114}`

Figures:
- Smoke latent space: [multimodal_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/multimodal_pca.png) and [multimodal_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space/multimodal_tsne.png)
- 50-epoch latent space: [multimodal_pca.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep/multimodal_pca.png) and [multimodal_tsne.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/latent_space_50ep/multimodal_tsne.png)

## Grad-CAM

- Both runs use the same refined visualization pipeline: MRI-derived brain masking, percentile thresholding, Gaussian smoothing, consistent cropping, and aligned MRI/PET overlays.
- The Grad-CAM outputs remain exploratory and should not be read as clinical anatomical evidence.
- Smoke selection manifest: `{'selected_subjects': [{'subject_id': 'sub-15', 'true_label': 'Control', 'predicted_label': 'Control', 'confidence': 0.4094749093055725, 'selection_reason': 'correct_prediction', 'figure_png': 'reports/figures/explainability/sub-15_multimodal_gradcam.png', 'figure_svg': 'reports/figures/explainability/sub-15_multimodal_gradcam.svg'}, {'subject_id': 'sub-20', 'true_label': 'Control', 'predicted_label': 'Control', 'confidence': 0.40909186005592346, 'selection_reason': 'correct_prediction', 'figure_png': 'reports/figures/explainability/sub-20_multimodal_gradcam.png', 'figure_svg': 'reports/figures/explainability/sub-20_multimodal_gradcam.svg'}, {'subject_id': 'sub-10', 'true_label': 'Control', 'predicted_label': 'Control', 'confidence': 0.4089857339859009, 'selection_reason': 'correct_prediction', 'figure_png': 'reports/figures/explainability/sub-10_multimodal_gradcam.png', 'figure_svg': 'reports/figures/explainability/sub-10_multimodal_gradcam.svg'}]}`
- 50-epoch selection manifest: `{'selected_subjects': [{'subject_id': 'sub-13', 'true_label': 'Control', 'predicted_label': 'Control', 'confidence': 0.9517098069190979, 'selection_reason': 'correct_prediction', 'figure_png': 'reports/figures/explainability_50ep/sub-13_multimodal_gradcam.png', 'figure_svg': 'reports/figures/explainability_50ep/sub-13_multimodal_gradcam.svg'}, {'subject_id': 'sub-09', 'true_label': 'Control', 'predicted_label': 'Control', 'confidence': 0.9391923546791077, 'selection_reason': 'correct_prediction', 'figure_png': 'reports/figures/explainability_50ep/sub-09_multimodal_gradcam.png', 'figure_svg': 'reports/figures/explainability_50ep/sub-09_multimodal_gradcam.svg'}, {'subject_id': 'sub-16', 'true_label': 'Control', 'predicted_label': 'Control', 'confidence': 0.9334123730659485, 'selection_reason': 'correct_prediction', 'figure_png': 'reports/figures/explainability_50ep/sub-16_multimodal_gradcam.png', 'figure_svg': 'reports/figures/explainability_50ep/sub-16_multimodal_gradcam.svg'}]}`

Figures:
- Smoke: [atlas_region_summary.json](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/atlas_region_summary.json)
- Smoke: [figure_selection_manifest.json](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/figure_selection_manifest.json)
- Smoke: [sub-10_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-10_multimodal_gradcam.png)
- Smoke: [sub-10_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-10_multimodal_gradcam.svg)
- Smoke: [sub-15_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-15_multimodal_gradcam.png)
- Smoke: [sub-15_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-15_multimodal_gradcam.svg)
- Smoke: [sub-20_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-20_multimodal_gradcam.png)
- Smoke: [sub-20_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability/sub-20_multimodal_gradcam.svg)
- 50-epoch: [atlas_region_summary.json](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/atlas_region_summary.json)
- 50-epoch: [figure_selection_manifest.json](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/figure_selection_manifest.json)
- 50-epoch: [sub-09_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-09_multimodal_gradcam.png)
- 50-epoch: [sub-09_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-09_multimodal_gradcam.svg)
- 50-epoch: [sub-13_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-13_multimodal_gradcam.png)
- 50-epoch: [sub-13_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-13_multimodal_gradcam.svg)
- 50-epoch: [sub-16_multimodal_gradcam.png](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-16_multimodal_gradcam.png)
- 50-epoch: [sub-16_multimodal_gradcam.svg](/home/kevin/Projects/python/neurodegenerative-pet-mri-ai/reports/figures/explainability_50ep/sub-16_multimodal_gradcam.svg)

## Limitations

- Only one AD subject is available in ds007561, so val/test do not include AD under the current fixed split.
- Any visual separation in PCA or t-SNE should be interpreted as exploratory structure, not clinical discrimination.
- The Grad-CAM overlays are useful for qualitative inspection of model attention, not for anatomical claims.
