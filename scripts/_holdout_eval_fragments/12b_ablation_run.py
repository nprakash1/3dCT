combos = list(product([0, 1], [0, 1], [0, 1]))
if SKIP_ALL_OFF:
    combos = [c for c in combos if c != (0, 0, 0)]
if QUICK_ABLATION:
    combos = [c for c in combos if c[0] == 1]

print('Holdout-eval ablation | %d combos | epochs<=%d' % (len(combos), ABLATION_EPOCHS))
print('Train CE: PROTO_TRAIN | Eval: random 1-of-%d holdout sents/class x %d passes' % (
    N_EVAL_SENTENCES, N_EVAL_PASSES))
print('combos (CE,Mag,SupCon):', combos)

rows = []
for run_i, (ce, mag, sup) in enumerate(combos):
    tag = 'ce%d_mag%d_supcon%d' % (ce, mag, sup)
    print('')
    print('=' * 60)
    print('[%d/%d] %s' % (run_i + 1, len(combos), tag))
    print('=' * 60)
    seed = ABLATION_SEED + run_i
    t0 = time.time()
    model, state, best_tune, best_ep, epochs_ran = _train_one(
        bool(ce), bool(mag), bool(sup), seed)
    tune_m = evaluate_holdout(model, 'tune', base_seed=seed + 999)
    test_m = evaluate_holdout(model, 'test', base_seed=seed + 1999) if RUN_TEST_EACH else None
    elapsed = time.time() - t0

    row = dict(
        ce=ce, mag=mag, supcon=sup, tag=tag, seed=seed,
        best_epoch=best_ep, epochs_ran=epochs_ran, seconds=round(elapsed, 1),
        tune_macro_f1=tune_m['macro_f1'], tune_macro_f1_std=tune_m['macro_f1_std'],
        tune_f1_worsened=tune_m['f1_worsened'],
        tune_f1_stable=tune_m['f1_stable'],
        tune_f1_improved=tune_m['f1_improved'],
        eval_paradigm='holdout_random_1of5',
    )
    if test_m is not None:
        row.update(
            test_macro_f1=test_m['macro_f1'],
            test_macro_f1_std=test_m['macro_f1_std'],
            test_f1_worsened=test_m['f1_worsened'],
            test_f1_stable=test_m['f1_stable'],
            test_f1_improved=test_m['f1_improved'],
        )
    rows.append(row)

    path = '%s/holdout_ablation_%s_seed%d_%s.pt' % (ABL_DIR, tag, seed, ts)
    torch.save(dict(model=state, row=row, findings=FINDINGS, classes=CLASSES), path)
    print('  saved', path)
    msg = '  tuneHoldoutF1=%.3f+/-%.3f' % (tune_m['macro_f1'], tune_m['macro_f1_std'])
    if test_m is not None:
        msg += '  testHoldoutF1=%.3f+/-%.3f' % (test_m['macro_f1'], test_m['macro_f1_std'])
    msg += '  |  %.1f min' % (elapsed / 60.0)
    print(msg)

df = pd.DataFrame(rows)
sort_col = 'test_macro_f1' if 'test_macro_f1' in df.columns else 'tune_macro_f1'
df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
print('')
print('=' * 60)
print('HOLDOUT-EVAL ABLATION SUMMARY (sorted by %s)' % sort_col)
print('=' * 60)
show = [c for c in [
    'ce', 'mag', 'supcon', 'tune_macro_f1', 'test_macro_f1',
    'test_macro_f1_std', 'test_f1_worsened', 'test_f1_stable', 'test_f1_improved',
    'best_epoch', 'epochs_ran', 'seconds',
] if c in df.columns]
try:
    display(df[show])
except NameError:
    print(df[show].to_string(index=False))

csv_path = '%s/holdout_loss_ablation_%s.csv' % (ABL_DIR, ts)
df.to_csv(csv_path, index=False)
print('wrote', csv_path)
ABLATION_DF = df
print('Done. ABLATION_DF ready. Eval never used train CE prototype strings.')
