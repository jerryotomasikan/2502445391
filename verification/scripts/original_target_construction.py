import time as _cell_time
_cell_started = _cell_time.perf_counter()
print('▶ SEL 2/11 DIMULAI — Memuat dan mengaudit dataset', flush=True)
try:
    def canonical_column(name):
        s = str(name).strip().lower()
        for ch in ['(', ')', '[', ']', '{', '}', ',', '^', '%', '/', '\\', '.', '-', ':', ';', '#']:
            s = s.replace(ch, ' ')
        return '_'.join(s.split())

    def locate_dataset(spec, explicit=None):
        candidates = []
        if explicit is not None:
            candidates.append(Path(explicit).expanduser())
        env_path = os.environ.get('PAPER_DATA_PATH') or os.environ.get('WINE_DATA_PATH')
        if env_path:
            candidates.append(Path(env_path).expanduser())
        fname = spec['file']
        cwd = Path.cwd()
        for base in [DATASET_DIR, PROJECT_ROOT / 'Dataset', PROJECT_ROOT / 'data',
                     cwd / 'Dataset', cwd / 'data', cwd, cwd.parent / 'Dataset']:
            candidates.append(base / fname)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        checked = '\n'.join(f'  - {p}' for p in candidates)
        raise FileNotFoundError(f'{fname} tidak ditemukan. Path yang diperiksa:\n' + checked)

    def _resolve_name(requested, columns):
        stripped = {str(c).strip(): c for c in columns}
        canon = {canonical_column(c): c for c in columns}
        if requested in stripped:
            return stripped[requested]
        if str(requested).strip() in stripped:
            return stripped[str(requested).strip()]
        if canonical_column(requested) in canon:
            return canon[canonical_column(requested)]
        raise KeyError(f'Kolom {requested!r} tidak ada. Tersedia: {list(columns)}')

    def read_dataset_frame(path, spec):
        frame = pd.read_csv(path, sep=spec.get('sep', ','))
        if frame.shape[1] == 1:
            frame = pd.read_csv(path, sep=None, engine='python')
        target_src = _resolve_name(spec['target'], frame.columns)
        drop_src = [_resolve_name(d, frame.columns) for d in spec.get('drop', [])]
        keep = [c for c in frame.columns if c not in set(drop_src)]
        frame = frame[keep].copy()
        rename = {c: canonical_column(c) for c in frame.columns}
        frame = frame.rename(columns=rename)
        target_col = rename[target_src]
        feature_cols = [c for c in frame.columns if c != target_col]
        if not feature_cols:
            raise ValueError('Tidak ada kolom fitur setelah drop/target.')
        return frame[feature_cols + [target_col]].copy(), feature_cols, target_col

    def _stratify_key(levels, min_count):
        """Gabung level ordinal dengan < min_count observasi ke tetangga terdekat.
        Menjaga StratifiedKFold(CV_SPLITS) valid. Untuk wine-red tidak ada level
        yang tergabung (perilaku identik paper)."""
        levels = np.asarray(levels)
        order = sorted(np.unique(levels).tolist())
        counts = {v: int(np.sum(levels == v)) for v in order}
        mapping = {v: v for v in order}
        changed = True
        while changed and len(set(mapping.values())) > 2:
            changed = False
            groups = sorted(set(mapping.values()))
            sizes = {g: sum(counts[v] for v in order if mapping[v] == g) for g in groups}
            small = [g for g in groups if sizes[g] < min_count]
            if not small:
                break
            g = small[0]
            gi = groups.index(g)
            neighbour = groups[gi + 1] if gi == 0 else groups[gi - 1]
            for v in order:
                if mapping[v] == g:
                    mapping[v] = neighbour
            changed = True
        remap = {g: i for i, g in enumerate(sorted(set(mapping.values())))}
        return np.array([remap[mapping[v]] for v in levels], dtype=np.int64), mapping

    def build_targets(target_values, spec):
        """Kembalikan (ordinal_int_consecutive, binary_int, meta).
        Definisi target adalah definisi tugas penelitian, bukan transform yang
        dipelajari; dihitung dari data bersih penuh, sama seperti 'quality >= 6'."""
        raw = np.asarray(target_values, dtype=float)
        cfg = spec['ordinal']
        meta = {'ordinal_kind': cfg['kind']}
        if cfg['kind'] == 'native':
            q = np.rint(raw).astype(np.int64)
            if not np.allclose(raw, q, atol=1e-9):
                raise ValueError('ordinal="native" tetapi target bukan bilangan bulat.')
            _, mapping = _stratify_key(q, RARE_LEVEL_MIN)
            merged = {int(k): int(v) for k, v in mapping.items() if k != v}
            if merged:
                print(f'  ⚠ Level ordinal langka digabung untuk menjaga CV: {merged}', flush=True)
            # pertahankan nilai grade asli (mis. wine 3..8); hanya level yang
            # digabung yang berpindah ke tetangganya.
            ordinal = np.array([int(mapping[int(v)]) for v in q], dtype=np.int64)
            meta['native_levels'] = sorted(np.unique(q).tolist())
            meta['merged_levels'] = merged
            # Penggabungan level di TENGAH dapat menyisakan lubang (mis. {1,2,4,5}).
            # Model ordinal kumulatif mensyaratkan level runut, jadi petakan ulang
            # sambil mempertahankan urutan. Untuk wine tidak ada lubang sehingga
            # grade 3..8 dipertahankan apa adanya.
            survivors = sorted(np.unique(ordinal).tolist())
            if survivors != list(range(survivors[0], survivors[-1] + 1)):
                renumber = {v: survivors[0] + i for i, v in enumerate(survivors)}
                print(f'  ⚠ Level tersisa tidak runut {survivors}; dipetakan ulang: {renumber}', flush=True)
                ordinal = np.array([renumber[int(v)] for v in ordinal], dtype=np.int64)
                meta['renumbered_levels'] = {int(k): int(v) for k, v in renumber.items()}
        else:
            bins = int(cfg['bins'])
            if cfg['strategy'] == 'quantile':
                codes = pd.qcut(raw, bins, labels=False, duplicates='drop')
            else:
                codes = pd.cut(raw, bins, labels=False, include_lowest=True)
            codes = np.asarray(codes, dtype=float)
            if np.isnan(codes).any():
                raise ValueError('Binning menghasilkan NaN; periksa target.')
            uniq = sorted(np.unique(codes).tolist())
            relabel = {u: i for i, u in enumerate(uniq)}
            ordinal = np.array([relabel[c] for c in codes], dtype=np.int64)
            # gabung bin langka (equal-width bisa menghasilkan bin sangat tipis)
            key2, mapping2 = _stratify_key(ordinal, RARE_LEVEL_MIN)
            if len(np.unique(key2)) != len(np.unique(ordinal)):
                merged2 = {int(k): int(v) for k, v in mapping2.items() if k != v}
                print(f'  ⚠ Bin ordinal langka digabung: {merged2}', flush=True)
                ordinal = key2
            meta['bins_requested'] = bins
            meta['bins_effective'] = int(len(np.unique(ordinal)))

        rule = spec['binary']
        if rule.startswith('ge:'):
            thr = float(rule.split(':', 1)[1])
            binary = (raw >= thr).astype(np.int64)
            meta['binary_rule'] = f'target >= {thr:g}'
        elif rule == 'median':
            med = float(np.median(raw))
            binary = (raw > med).astype(np.int64)
            meta['binary_rule'] = f'target > median ({med:.6g})'
        else:
            raise ValueError(f'binary rule tidak dikenal: {rule}')
        if len(np.unique(binary)) < 2:
            raise ValueError('Target biner hanya memiliki satu kelas.')
        return ordinal, binary, meta

    # --- muat dataset terpilih ---
    DATA_FILE = locate_dataset(DATASET_SPEC, DATA_PATH)
    SOURCE_SHA256 = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    frame_full, FEATURE_COLUMNS, TARGET_RAW_COLUMN = read_dataset_frame(DATA_FILE, DATASET_SPEC)

    for col in frame_full.columns:
        frame_full[col] = pd.to_numeric(frame_full[col], errors='coerce')
    if frame_full.isna().any().any():
        n_bad = int(frame_full.isna().sum().sum())
        raise ValueError(f'Ditemukan {n_bad} nilai missing/non-numerik pada fitur/target.')
    if not np.isfinite(frame_full[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError('Fitur mengandung nilai non-finite.')

    subsample_n = DATASET_SPEC.get('subsample')
    subsample_applied = 0
    if subsample_n and len(frame_full) > subsample_n:
        frame_full = frame_full.sample(n=int(subsample_n), random_state=SEED).sort_index().reset_index(drop=True)
        subsample_applied = int(subsample_n)

    df_raw = frame_full.reset_index(drop=True).copy()
    ORD_RAW, BIN_RAW, TARGET_META = build_targets(df_raw[TARGET_RAW_COLUMN].to_numpy(), DATASET_SPEC)
    df_raw = df_raw.assign(ordinal_level=ORD_RAW, binary_target=BIN_RAW)

    strict_conflict = (DATASET_SPEC['ordinal']['kind'] == 'native')
    conflicts = df_raw.groupby(FEATURE_COLUMNS, dropna=False)['ordinal_level'].nunique()
    conflict_count = int((conflicts > 1).sum())
    if conflict_count and strict_conflict:
        raise ValueError(f'Ada {conflict_count} vektor fitur identik dengan label ordinal berbeda; audit manual diperlukan.')
    if conflict_count:
        print(f'  ⚠ {conflict_count} grup fitur identik berlabel berbeda (target kontinu); baris pertama dipertahankan.', flush=True)

    # III.B: deduplikasi SEBELUM split adalah default protokol. dedup=False
    # sengaja MEMBIARKAN vektor identik menyeberang batas train/test — dipakai
    # hanya untuk mengukur besarnya inflasi akibat kebocoran duplikat (A/B).
    DEDUP = bool(DATASET_SPEC.get('dedup', True))
    duplicate_mask = df_raw.duplicated(subset=FEATURE_COLUMNS, keep='first')
    if DEDUP:
        df = df_raw.loc[~duplicate_mask].reset_index(drop=True)
    else:
        print(f'  ⚠ DEDUPLIKASI DIMATIKAN — {int(duplicate_mask.sum())} vektor identik '
              f'dibiarkan dan dapat menyeberang train/test (lengan A/B kebocoran).', flush=True)
        df = df_raw.reset_index(drop=True)

    TARGET_COLUMN = 'ordinal_level'
    Y_QUALITY = df['ordinal_level'].to_numpy(dtype=np.int64)
    Y_BINARY = df['binary_target'].to_numpy(dtype=np.int64)
    X_ALL = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    ORDINAL_LEVELS = np.array(sorted(np.unique(Y_QUALITY).tolist()), dtype=np.int64)
    ORDINAL_MIN = int(ORDINAL_LEVELS.min())
    ORDINAL_MAX = int(ORDINAL_LEVELS.max())
    if not np.array_equal(ORDINAL_LEVELS, np.arange(ORDINAL_MIN, ORDINAL_MAX + 1)):
        raise RuntimeError(f'Level ordinal tidak berurutan: {ORDINAL_LEVELS.tolist()}')
    if len(ORDINAL_LEVELS) < 2:
        raise RuntimeError('Target ordinal hanya memiliki satu level setelah pemrosesan.')

    canonical_bytes = df[FEATURE_COLUMNS + ['ordinal_level', 'binary_target']].to_csv(
        index=False, float_format='%.17g', lineterminator='\n').encode('utf-8')
    CLEAN_SHA256 = hashlib.sha256(canonical_bytes).hexdigest()

    DATA_AUDIT = {
        'dataset_id': DATASET_ID, 'spec': json.loads(json.dumps(DATASET_SPEC, default=str)),
        'source_path': str(DATA_FILE), 'source_sha256': SOURCE_SHA256,
        'clean_sha256': CLEAN_SHA256, 'rows_raw': int(len(df_raw)),
        'rows_clean': int(len(df)), 'duplicates_removed': int(duplicate_mask.sum()),
        'subsample_applied': subsample_applied, 'dedup': DEDUP,
        'conflicting_feature_groups': conflict_count,
        'target_column_source': TARGET_RAW_COLUMN, 'target_meta': TARGET_META,
        'ordinal_levels': [int(v) for v in ORDINAL_LEVELS],
        'ordinal_counts_clean': {str(k): int(v) for k, v in pd.Series(Y_QUALITY).value_counts().sort_index().items()},
        'binary_counts_clean': {str(k): int(v) for k, v in pd.Series(Y_BINARY).value_counts().sort_index().items()},
        'features': FEATURE_COLUMNS, 'n_features': len(FEATURE_COLUMNS),
    }

    print(f'Dataset          : {DATASET_ID}  <-  {DATA_FILE.name}')
    print(f'Fitur ({len(FEATURE_COLUMNS)})       : {FEATURE_COLUMNS}')
    print(f'Target sumber     : {TARGET_RAW_COLUMN}  | {TARGET_META.get("binary_rule")}')
    if subsample_applied:
        print(f'Subsample        : {subsample_applied} baris (seed {SEED})')
    print(f'Bentuk mentah    : {df_raw.shape}')
    print(f'Duplikat dibuang : {int(duplicate_mask.sum())}')
    print(f'Bentuk bersih    : {df.shape}')
    print(f'Level ordinal    : {ORDINAL_LEVELS.tolist()}  ({TARGET_META.get("ordinal_kind")})')
    print('Distribusi ordinal:', DATA_AUDIT['ordinal_counts_clean'])
    print('Target biner      :', DATA_AUDIT['binary_counts_clean'], f'(0/1 via {TARGET_META.get("binary_rule")})')
    print(f'Clean SHA256      : {CLEAN_SHA256}')
    display(df.head())
except KeyboardInterrupt:
    print('⏸ SEL 2/11 DIHENTIKAN OLEH PENGGUNA — Memuat dan mengaudit dataset', flush=True)
    raise
except Exception as _cell_error:
    print(f'❌ SEL 2/11 GAGAL — {type(_cell_error).__name__}: {_cell_error}', flush=True)
    raise
else:
    print(f'✅ SEL 2/11 SELESAI — Memuat dan mengaudit dataset ({_cell_time.perf_counter() - _cell_started:.1f} detik)', flush=True)
