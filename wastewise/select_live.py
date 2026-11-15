"""Select deployed backbone by validation macro F1 and publish honest evidence."""
from .common import ROOT, LABELS, read_json, save_json, digest


def main():
    candidates = []
    manifests = set()
    for name in ('selective', 'selective_efficientnet'):
        run = ROOT / 'artifacts/runs' / name
        if not (run / 'metrics.json').exists():
            continue
        metrics, metadata = read_json(run / 'metrics.json'), read_json(run / 'metadata.json')
        for filename, expected in metadata['sha256'].items():
            if digest(run / filename) != expected:
                raise ValueError(f'Corrupt experiment artifact: {name}/{filename}')
        manifests.add(metadata['manifest_sha256'])
        candidates.append(dict(run=run.relative_to(ROOT).as_posix(),
                               validation_macro_f1=metrics['validation']['macro_f1'],
                               head=metrics['selected']))
    if not candidates:
        raise ValueError('Train a selective classifier first.')
    if len(manifests) != 1:
        raise ValueError('Candidate manifests differ; validation scores are not directly comparable.')
    comparison_manifest = next(iter(manifests))
    # A later data expansion is comparable only using the frozen baseline scored
    # on that same expanded validation set, never its earlier easier validation.
    for expanded_name, baseline_name in (('selective_organic', 'selective_efficientnet'),
                                         ('selective_everyday', 'selective_organic')):
        expanded = ROOT / 'artifacts/runs' / expanded_name
        if not (expanded / 'metrics.json').exists() or not (expanded / 'metadata.json').exists():
            continue
        added_metrics = read_json(expanded / 'metrics.json')
        added_metadata = read_json(expanded / 'metadata.json')
        decision = read_json(expanded / 'selection_decision.json')
        baseline = ROOT / 'artifacts/runs' / baseline_name
        if max(candidates, key=lambda candidate: candidate['validation_macro_f1'])['run'] != baseline.relative_to(ROOT).as_posix():
            raise ValueError('Expanded experiment must compare against the previously selected baseline')
        baseline_metadata = read_json(baseline / 'metadata.json')
        if decision['baseline_head_sha256'] != baseline_metadata['sha256']['head.npz']:
            raise ValueError('Expanded comparison refers to a different frozen baseline')
        if decision['manifest_sha256'] != added_metadata['manifest_sha256']:
            raise ValueError('Expanded comparison manifest mismatch')
        for filename, expected in added_metadata['sha256'].items():
            if digest(expanded / filename) != expected:
                raise ValueError(f'Corrupt expanded experiment artifact: {filename}')
        if abs(added_metrics['validation']['macro_f1']-decision['new_validation_macro_f1']) > 1e-10:
            raise ValueError('Validation selection record mismatch')
        if abs(added_metrics['frozen_baseline']['validation']['macro_f1']-decision['baseline_validation_macro_f1']) > 1e-10:
            raise ValueError('Frozen validation record mismatch')
        candidates = [dict(run=baseline.relative_to(ROOT).as_posix(),
                           validation_macro_f1=decision['baseline_validation_macro_f1'], head=baseline_metadata['selected']),
                      dict(run=expanded.relative_to(ROOT).as_posix(),
                           validation_macro_f1=decision['new_validation_macro_f1'], head=added_metadata['selected'])]
        comparison_manifest = decision['manifest_sha256']
    selected = max(candidates, key=lambda candidate: candidate['validation_macro_f1'])
    # The first candidate is the frozen deployed baseline, including when the
    # new experiment loses. Never expose a rejected experiment as "previous".
    previous_run = candidates[0]['run']
    selection = dict(**selected, candidates=candidates, previous_run=previous_run,
                     comparison_manifest_sha256=comparison_manifest,
                     selection_rule='Maximum validation macro F1; test scores not used in backbone/head selection.')
    # Only after selection is fixed do we report test outcomes.
    metrics = read_json(ROOT / selected['run'] / 'metrics.json')
    save_json(ROOT / 'artifacts/live_metrics.json', metrics)
    test = metrics['test']
    original = read_json(ROOT / 'artifacts/baseline_mobilenet.json')
    previous = read_json(ROOT / 'artifacts/runs/mobilenet/metrics.json')['test']
    frozen = read_json(ROOT / 'artifacts/runs/selective_efficientnet/metrics.json')['test']
    organic_metrics = ROOT / 'artifacts/runs/selective_organic/metrics.json'
    organic_previous = read_json(organic_metrics)['test'] if organic_metrics.exists() else None
    lines = ['# Live detector experiment results', '',
             f"Selected experiment: `{selected['run']}`; head: `{selected['head']}`.", '',
             'Backbone and head selected using validation macro F1 only. The historical test set was previously examined during development; these are retrospective results, not a pristine field benchmark.', '',
             '| Classifier | Historical accuracy | Macro F1 |', '| --- | ---: | ---: |',
             f"| Original MobileNet | {original['accuracy']:.2%} | {original['macro_f1']:.3f} |",
             f"| Previous transfer model | {previous['accuracy']:.2%} | {previous['macro_f1']:.3f} |",
             f"| EfficientNet before produce expansion | {frozen['accuracy']:.2%} | {frozen['macro_f1']:.3f} |",
             *([f"| Produce-expanded previous model | {organic_previous['accuracy']:.2%} | {organic_previous['macro_f1']:.3f} |"] if organic_previous else []),
             f"| Selected classifier, all images | {test['accuracy']:.2%} | {test['macro_f1']:.3f} |", '',
             f"Historical test images: **{test['n']}**. The selected classifier's rejection gate accepts **{test['selective_coverage']:.2%}** of these; accepted accuracy is **{test['selective_accepted_accuracy']:.2%}**. Rejections are not removed from the all-image accuracy above.", '',
             'These selective scores include confidence, class margin and training-neighbor support. They exclude foreground localization, low-detail crop filtering and temporal confirmation. They do not measure live-camera accuracy.', '',
             '| True class | Total | Accepted | Correct among accepted |', '| --- | ---: | ---: | ---: |']
    for label in LABELS:
        values = test['selective_per_class'][label]
        lines.append(f"| {label} | {values['n']} | {values['accepted']} | {values['correct_accepted']} |")
    lines += ['', 'Class-specific acceptance matters: a high accepted accuracy can hide a class that is mostly rejected. Thresholds were fitted on whole-image validation; foreground crops introduce a different input distribution.', '',
              '## Validation selection', '', '| Experiment | Validation macro F1 | Head |', '| --- | ---: | --- |']
    for candidate in candidates:
        lines.append(f"| {candidate['run']} | {candidate['validation_macro_f1']:.5f} | {candidate['head']} |")
    if 'organic_holdout' in metrics:
        organic = metrics['organic_holdout']
        food = metrics['external_test']
        lines += ['', '## Additional organic data and untouched source-folder holdout', '',
                  'Fruits-360: 840 sampled images; 498 retained after duplicate screening, with 306 added to training, 102 to validation and 90 peach/plum images held out by source folder. Attribution and CC BY-SA 4.0 terms are recorded in `artifacts/organic_sources.json`. Produce is an organic-material proxy, not a measurement of moisture or proof that food should be discarded.', '',
                  f"New peach/plum holdout: {organic['accuracy']:.2%} raw organic recall across {organic['n']} images; selective coverage {organic['selective_coverage']:.2%}. This is one class, not four-class accuracy.", '',
                  f"Existing RealWaste organic holdout: {food['accuracy']:.2%} raw organic recall across {food['n']} images; selective coverage {food['selective_coverage']:.2%}.", '',
                  'The candidate and frozen baseline were compared on exactly the same expanded validation set. The deployment recommendation was written before holdout/test evaluation. Historical results are still exploratory because the data expansion followed historical error analysis.', '']
    everyday = ROOT / 'artifacts/runs/selective_everyday/metrics.json'
    if everyday.exists():
        expansion = read_json(everyday)
        provenance = read_json(ROOT / 'artifacts/everyday_sources.json')
        held = expansion['everyday_holdout']
        before = expansion['frozen_baseline']['everyday_holdout']
        lines += ['', '## Everyday-object expansion', '',
            f"Added {provenance['retained']} audited examples from TACO v1.0 and RealWaste vegetation. "
            f"Added training images: {sum(provenance['splits']['train'].values())}; validation: {sum(provenance['splits']['validation'].values())}; separate everyday holdout: {held['n']}.", '',
            'Includes bottles, cans, paper/cardboard, packaging film/wrappers, straws, foam and vegetation. TACO uses isolated annotation crops; only one crop per source image is retained. Near-duplicate overlap with existing data is excluded before split assignment. Old images and assignments remain intact.', '',
            '| Model | New everyday holdout accuracy | Macro F1 |', '| --- | ---: | ---: |',
            f"| Frozen previous model | {before['accuracy']:.2%} | {before['macro_f1_present_classes']:.3f} |",
            f"| Retrained everyday model | {held['accuracy']:.2%} | {held['macro_f1_present_classes']:.3f} |", '',
            'Everyday holdout macro F1 averages only its represented classes: '+', '.join(held['represented_classes'])+'. '
            'It is a source-image test, not a live-camera benchmark. Promotion uses common validation data, never the holdout scores above.', '',
            f"**Tradeoff:** historical 922-image accuracy changes from {expansion['frozen_baseline']['test']['accuracy']:.2%} to {expansion['test']['accuracy']:.2%}. "
            'The default is chosen for the broader validation distribution. Run `demo.py --model previous` to compare the preserved earlier bundle.', '',
            'Source licenses, mappings, hashes, exclusions and download failures are recorded in `artifacts/everyday_sources.json`. Material labels do not prove local recycling eligibility. Unrepresented items can still be misclassified; human review remains useful.', '']
        if held.get('by_source_category'):
            item_lines=['# Results by everyday item type', '',
                'These are held-out source images, not a live-camera accuracy test. No source image contributes a crop to multiple splits. Small counts make these percentages uncertain.', '',
                '| Item type | Holdout images | Frozen previous model | Retrained model |',
                '| --- | ---: | ---: | ---: |']
            for category, values in sorted(held['by_source_category'].items()):
                old=before['by_source_category'][category]
                item_lines.append(f"| {category} | {values['n']} | {old['accuracy']:.1%} | {values['accuracy']:.1%} |")
            item_lines += ['', 'A missing item type has no held-out examples in this subset; it does not imply perfect recognition. '
                           'Model choice is fixed using validation before these results are calculated.', '',
                           '[Dataset mappings and limits](DATA_COVERAGE.md) · [Overall results](LIVE_RESULTS.md)', '']
            (ROOT/'docs/ITEM_RESULTS.md').write_text('\n'.join(item_lines),encoding='utf-8')
    lines += ['', '## Camera behavior', '',
              'Fixed-camera background calibration, foreground bounding boxes, crop-only inference, uncertainty rejection and three-frame label confirmation are integrated in the OpenCV program. B recalibrates an empty view; Q exits. Boxes are classical foreground proposals, not trained semantic detections.', '',
              'The actual webcam acquisition check processed 90 frames, saved none, and reported 58 no-object frames after calibration. This was not a labeled field evaluation. Regression tests cover synthetic empty walls, stationary objects, removal, lighting changes, multiple objects, crop-only classification and simulated Arduino gating.', '',
              'The original model is unchanged. No images were silently captured for training and no physical motors were activated.', '']
    lines += ['## Evidence-backed resume bullet', '',
              f"- Built an OpenCV waste-segregation pipeline with foreground localization and uncertainty rejection; compared pretrained CNN backbones and classifier configurations, improving retrospective accuracy from {original['accuracy']:.2%} to {test['accuracy']:.2%} and macro F1 from {original['macro_f1']:.3f} to {test['macro_f1']:.3f} on {test['n']} audited images.", '',
              'This describes classifier metrics on the historical image set, not measured localization accuracy, live accuracy or a novel research architecture.', '']
    (ROOT / 'docs/LIVE_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    # Publish the pointer last, using an atomic replacement after evidence exists.
    temporary = ROOT / 'artifacts/live_model.json.part'
    save_json(temporary, selection)
    temporary.replace(ROOT / 'artifacts/live_model.json')
    print(selected)


if __name__ == '__main__':
    main()
