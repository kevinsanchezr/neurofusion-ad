# TODO — Proyecto multimodal MRI + FDG-PET / ADNI

Fecha de sincronización: 10 de septiembre de 2026.

## Estado verificado

- [x] Cohorte auditada: 54 sujetos, 18 CN + 18 MCI + 18 AD; un MRI y un FDG-PET por sujeto.
- [x] Conversión nativa: 54 MRI DICOM y 54 PET (53 ECAT7 + PET-DICOM BRAIN 2 de `153_S_4172`).
- [x] C1/C1-R1/C1-R2: herramientas, SynthStrip y `standard_synra` validados.
- [x] C2: 54/54 pares procesados en MNI152NLin2009cAsym, 2 mm, RAS, 97×115×97.
- [x] QC C2: 51 PASS, 1 WARNING_ACCEPTED, 2 WARNING, 0 FAIL; ninguno excluido.
- [x] Excepciones documentadas: `009_S_1199`, `036_S_1001`, `153_S_4172`.
- [x] C2 congelado: 108 NIfTI con hashes finales y sin bits de escritura.
- [x] Pipeline C2 protegido por `FROZEN.json`; el informe STOPPED se conserva solo como historial.
- [x] Stage D de preparación: nested CV 3×3×3, cuatro variantes materializadas, índices MRI/PET/multimodales, QC y provenance completos.
- [ ] Entrenamiento ADNI: no iniciado.

Fuentes vigentes: `adni_c2_manifest.csv`, `adni_stage_c2_qc.csv`, `adni_stage_c2_complete_report.md`, `FROZEN.json` y `frozen_pairs_sha256.csv`. El antiguo `adni_stage_c2_preprocessing_report.md` no representa el estado actual.

## Objetivo científico

Evaluar si la fusión MRI T1 + FDG-PET mejora clasificación CN/MCI/AD frente a MRI-only y PET-only con los mismos sujetos, splits y presupuesto. Es un estudio exploratorio, no predicción longitudinal ni sistema clínico validado. `ds007561` y sus resultados permanecen separados.

## P0 — cierre C2

- [x] Conciliar 54 filas, 54 IDs, 18/18/18, rutas y estados.
- [x] Confirmar 54 MRI MNI + 54 PET MNI finitos y no vacíos.
- [x] Confirmar forma 97×115×97, 2 mm, RAS.
- [x] Adjudicar `036_S_1001` mediante regla universal C2-PET-QC-1.
- [x] Mantener WARNING elegibles de `009_S_1199` y `153_S_4172`.
- [x] Congelar pares y registrar SHA-256.
- [x] Evitar sobrescritura mediante permisos y guard del pipeline.

## P1 — Stage D de preparación, completado sin entrenamiento

Plan normativo: `reports/adni_stage_d_plan.md`.

- [x] Nested CV sujeto-nivel 3 outer × 3 inner, repetida 3 veces.
- [x] Tres repeticiones; semillas bloqueadas: 20260910, 20261007 y 20261103.
- [x] Balanced accuracy multiclase primaria y macro-F1 secundaria.
- [x] PET robust relative y whole-brain mean materializados; SUVR bloqueado hasta QC anatómico.
- [x] Pons/cerebelo aprobados solo como candidatos anatómicos; aún no evaluados ni seleccionados.
- [x] MRI robust z-score y percentile [0,1] materializados como variantes no seleccionadas.
- [x] Los 54 sujetos se conservan; sensibilidad a |delta_days| y excepciones queda predefinida.
- [x] Entradas float32, 97×115×97, sin 128³.

Después de aprobación, pero no antes:

- [x] Índice sujeto-nivel generado desde el manifiesto congelado; validación clínica adicional queda antes de entrenamiento.
- [x] Splits deterministas 3×3×3 generados sin resultados de modelos.
- [x] Disjunción, balance, hashes y agrupamiento auditados: 27/27 PASS.
- [ ] Seleccionar/comparar normalizaciones exclusivamente dentro de inner CV cuando se autorice entrenamiento; outer test permanece ciego.
- [x] Entradas MRI-only, PET-only y multimodales indexadas; 216 NIfTI normalizados con hashes.
- [x] QC y reporte Stage D emitidos; ejecución detenida antes de entrenamiento.

## P2 — protocolo y entrenamiento futuro

- [ ] Fijar arquitectura, presupuesto, early stopping, búsqueda y semillas de inicialización.
- [ ] Usar folds idénticos para MRI-only, PET-only y multimodal.
- [ ] Ajustar toda estadística aprendida solo con training inner.
- [ ] Aplicar aumentos solo en training y sincronizados entre modalidades.
- [ ] Ejecutar smoke test y overfit técnico sobre training, sin presentarlo como resultado.
- [ ] Verificar VRAM de RTX 4050, batch pequeño, precisión mixta y acumulación.
- [ ] Autorizar por separado la campaña completa.
- [ ] Exportar predicciones OOF, métricas por fold, checkpoints, logs y hashes.

## P3 — análisis de sensibilidad e interpretabilidad

- [ ] Predefinir sensibilidad a |delta_days| ≤30 sin alterar análisis primario.
- [ ] Evaluar por separado `009_S_1199`, `036_S_1001` y `153_S_4172`.
- [ ] Examinar centro/escáner/protocolo/edad/sexo sin sobreajustar n=54.
- [ ] Elegir atlas MNI versionado y validar ROIs/cobertura.
- [ ] Generar atribuciones solo sobre sujetos retenidos y modelos outer válidos.
- [ ] No concatenar embeddings de folds distintos sin alineación justificada.
- [ ] Mantener conclusiones como asociaciones exploratorias, no biomarcadores causales.

## P4 — artículo y reproducibilidad

- [ ] Mantener el título de trabajo: “Evaluating Multimodal Deep Learning Fusion of T1-Weighted MRI and FDG-PET for Cognitive-State Classification”.
- [ ] Corregir balanced accuracy: media del recall de las tres clases.
- [ ] Describir conversión ECAT/PET-DICOM, SynthStrip, N4, motion correction, registros y QC reales.
- [ ] Elegir sede IEEE concreta antes de adaptar plantilla/requisitos.
- [ ] Verificar fuentes primarias, DOI, acknowledgements y condiciones ADNI.
- [ ] Completar resultados solo desde artefactos finales reproducibles.
- [ ] Documentar dependencias, versiones, plantillas, pesos, commits, splits y hashes.
- [ ] No publicar imágenes ADNI ni datos restringidos automáticamente.

## Puertas permanentes

Detener si cambia un hash congelado; hay sujeto/modalidad ausente o duplicado; aparece fuga; se ajusta con validation/test; cambian geometría/affine; hay NaN/Inf/vacío; falla cobertura SUVR; se necesita una regla individual; o un documento contradice la fuente de verdad.

## Próxima acción autorizable

Kevin revisa el reporte Stage D. La próxima autorización posible es validar ROI SUVR y/o definir el entrenamiento inner-CV; outer test seguirá ciego. No se ha entrenado ningún modelo.
