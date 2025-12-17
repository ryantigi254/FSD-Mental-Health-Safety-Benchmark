# Study A Gold Labels & Metrics Reorganization

## Summary

Study A gold label and metrics files have been reorganized for better structure and navigation.

## New Structure

### Data Files

**Before:**
```
data/openr1_psy_splits/
├── study_a_test.json
├── study_a_gold_diagnosis_labels.json
├── study_a_gold_labels_mapping.json
└── study_a_diagnosis_suggestions.json
```

**After:**
```
data/
├── openr1_psy_splits/
│   └── study_a_test.json (frozen test split)
└── study_a_gold/ (NEW)
    ├── gold_diagnosis_labels.json
    ├── gold_labels_mapping.json
    └── diagnosis_suggestions.json
```

### Scripts

**Before:**
```
scripts/
├── populate_gold_labels_from_openr1.py
├── verify_gold_labels_mapping.py
├── verify_id_matching.py
├── manual_label_study_a_gold.py
├── init_study_a_gold_labels.py
├── populate_remaining_labels.py
├── extract_diagnosis_from_openr1.py
├── calculate_study_a_metrics.py
└── extract_study_a_predictions.py
```

**After:**
```
scripts/
└── study_a/ (NEW)
    ├── gold_labels/
    │   ├── populate_from_openr1.py
    │   ├── verify_mapping.py
    │   ├── verify_id_matching.py
    │   ├── manual_label.py
    │   ├── init_labels.py
    │   ├── populate_remaining.py
    │   └── extract_suggestions.py
    └── metrics/
        ├── calculate_metrics.py
        └── extract_predictions.py
```

## Benefits

1. **Clear Separation**: Gold labels separated from frozen test splits
2. **Better Organization**: Scripts grouped by function (gold_labels vs metrics)
3. **Easier Navigation**: Logical directory structure
4. **Maintainability**: Related files grouped together

## Updated Code

All path references have been updated:

- `study_a_loader.py` - Updated to check new location first, fallback to old for compatibility
- All gold label scripts - Updated paths to `data/study_a_gold/`
- All metrics scripts - Updated import paths for new directory structure

## Documentation

- `data/study_a_gold/README.md` - Gold label files documentation
- `scripts/study_a/README.md` - Study A scripts overview
- `docs/data/STUDY_A_GOLD_LABELS_MAPPING.md` - Updated with new paths
- `docs/README.md` - New documentation index

## Migration Notes

- Old paths still work (backwards compatibility in `study_a_loader.py`)
- All scripts updated to use new paths
- Verification scripts tested and working

## Quick Reference

### Extract Gold Labels
```bash
python scripts/study_a/gold_labels/populate_from_openr1.py --force
```

### Verify ID Matching
```bash
python scripts/study_a/gold_labels/verify_id_matching.py
```

### Calculate Metrics
```bash
python scripts/study_a/metrics/calculate_metrics.py
```

