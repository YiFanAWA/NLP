import json, os, shutil, datetime
base = r'E:/new_test/TCMERE-master/TCMERE-master/TCMERE/TCMERE-main'
res_dir = os.path.join(base, 'resources', 'biorelex')
train_fp = os.path.join(res_dir, 'train.json')
dev_fp = os.path.join(res_dir, 'dev.json')
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
for fp in [train_fp, dev_fp]:
    if not os.path.exists(fp):
        print('Missing', fp)
        continue
    shutil.copy(fp, fp + f'.bak.{now}')
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data is list of samples
    for inst in data:
        # ensure entities is list
        ents = inst.get('entities', [])
        for cluster in ents:
            # add cluster-level is_mentioned
            cluster.setdefault('is_mentioned', True)
            # names is dict
            names = cluster.get('names', {})
            for alias, meta in list(names.items()):
                if isinstance(meta, dict):
                    meta.setdefault('is_mentioned', True)
                    # ensure mentions exist
                    if 'mentions' not in meta:
                        meta['mentions'] = []
                    # ensure label in name meta
                    meta.setdefault('label', cluster.get('label', ''))
                else:
                    # if name meta is not dict, replace with dict
                    names[alias] = {'is_mentioned': True, 'mentions': [], 'label': cluster.get('label', '')}
            cluster['names'] = names
        inst['entities'] = ents
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('Fixed', fp, '-> backup at', fp + f'.bak.{now}')
print('Done')
