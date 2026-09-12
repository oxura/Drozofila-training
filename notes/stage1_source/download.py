"""Download the exact upstream snapshot used in this experiment."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = '91bdd1e7dcf193f3e7ca5a8933497fcef63b7960'
FILES = ['Connectivity_783.parquet', 'Completeness_783.csv',
         'model.py', 'utils.py', 'Readme.md', 'LICENSE', 'environment.yml']


def main():
    manifest_path = ROOT / 'data/source_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'repository': 'https://github.com/philshiu/Drosophila_brain_model',
        'commit': COMMIT, 'files': []}
    known = {f['path']: f for f in manifest['files']}
    for name in FILES:
        relative = ('data/raw/' if name.endswith(('.csv', '.parquet')) else 'upstream/') + name
        path = ROOT / relative
        expected = known.get(relative, {}).get('sha256')
        if path.exists() and expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print('verified', relative)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{COMMIT}/{name}'
        temp = path.with_suffix(path.suffix + '.part')
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=90) as response, temp.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        if expected and digest.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        temp.replace(path)
        known[relative] = dict(path=relative, url=url, bytes=path.stat().st_size,
                               sha256=digest.hexdigest())
        manifest['files'] = list(known.values())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print('downloaded', relative)


if __name__ == '__main__':
    main()
