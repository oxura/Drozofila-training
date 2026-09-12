"""Reconstruct original large data from versioned, hash-checked binary chunks."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def restore():
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    for item in manifest['files']:
        target=ROOT/item['path']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
            print(f'Already verified: {item["path"]}');continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.part');digest=hashlib.sha256();total=0
        with temporary.open('wb') as output:
            for part in item['parts']:
                data=(ROOT/part['path']).read_bytes()
                assert len(data)==part['bytes'] and hashlib.sha256(data).hexdigest()==part['sha256'],part['path']
                output.write(data);digest.update(data);total+=len(data)
        assert total==item['bytes'] and digest.hexdigest()==item['sha256'],'Reconstruction checksum mismatch'
        temporary.replace(target)
        print(f'Restored and verified: {item["path"]} ({total} bytes)')


if __name__=='__main__':restore()
