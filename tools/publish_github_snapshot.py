"""Publish an immutable staged Git snapshot through the connected GitHub app.

The SDK directory must contain the runtime-provided library_hosted_apps.py.
Credentials are handled by that SDK and are never written into the repository.
Only immutable blob uploads run concurrently; tree/commit/ref updates are ordered.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--parent', required=True)
    p.add_argument('--expected-tree', required=True)
    p.add_argument('--message-file', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--sdk-dir', required=True)
    p.add_argument('--connector-id', required=True)
    p.add_argument('--repository', default='oxura/Drozofila-training')
    p.add_argument('--state-dir', default='/tmp/flybrain-publish')
    args = p.parse_args()
    sys.path.insert(0, args.sdk_dir)
    from library_hosted_apps import HostedAppsClient
    client = HostedAppsClient(); root = Path.cwd()
    state = Path(args.state_dir); state.mkdir(parents=True, exist_ok=True)

    def git(*parts):
        return subprocess.check_output(['git', *parts], cwd=root)

    def unpack(result):
        content = result.get('structuredContent')
        if content is None:
            texts = [item['text'] for item in result.get('content', []) if item.get('type') == 'text']
            assert len(texts) == 1, 'Expected one JSON result from the hosted app.'
            content = json.loads(texts[0])
        return content.get('result', content)

    def call(name, values):
        r = client.call_tool(args.connector_id, name, dict(repository_full_name=args.repository, **values))
        return unpack(r)

    def remote_head():
        r = client.call_tool(args.connector_id, 'github_fetch',
            {'url': f'https://api.github.com/repos/{args.repository}/git/ref/heads/main'})
        content = unpack(r)
        return json.loads(content['content'])['object']['sha']

    assert git('rev-parse', 'HEAD').decode().strip() == args.parent
    assert git('write-tree').decode().strip() == args.expected_tree
    assert remote_head() == args.parent, 'Remote advanced; reconcile before publishing.'
    entries = []; binary = {}; text_entries = []
    for line in Path(args.index).read_text().splitlines():
        meta, name = line.split('\t', 1); mode, sha, stage = meta.split()
        assert stage == '0' and mode in ('100644', '100755'), name
        data = git('cat-file', 'blob', sha)
        try:
            content = data.decode('utf-8') if len(data) <= 16384 else None
        except UnicodeDecodeError:
            content = None
        if content is None:
            binary[sha] = name
            entries.append(dict(path=name, mode=mode, type='blob', sha=sha))
        else:
            text_entries.append(dict(path=name, mode=mode, type='blob', content=content))
    cache_file = state / 'uploaded-blobs.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    lock = threading.Lock(); last_start = [0.]

    def upload(item):
        sha, name = item
        if sha in cache: return
        data = git('cat-file', 'blob', sha)
        with lock:
            delay = max(0., .85 - (time.monotonic() - last_start[0]))
            if delay: time.sleep(delay)
            last_start[0] = time.monotonic()
        result = call('github_create_blob', dict(encoding='base64', content=base64.b64encode(data).decode('ascii')))
        assert result['sha'] == sha, name
        with lock:
            cache[sha] = name
            temp = cache_file.with_suffix('.tmp')
            temp.write_text(json.dumps(cache)); temp.replace(cache_file)

    print(f'Snapshot: {len(entries) + len(text_entries)} paths; {len(binary)} unique binary/large blobs', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(upload, item) for item in binary.items()]
        for i, task in enumerate(as_completed(futures), 1):
            task.result()
            if i % 20 == 0 or i == len(futures): print(f'Blobs ready: {i}/{len(futures)}', flush=True)
    tree = git('rev-parse', args.parent + '^{tree}').decode().strip()
    batch = []; size = 0

    def write_tree(portion, base):
        return call('github_create_tree', dict(base_tree_sha=base, tree_elements=portion))['sha']

    for entry in entries + text_entries:
        amount = len(entry.get('content', '').encode())
        if batch and (len(batch) == 20 or size + amount > 100000):
            tree = write_tree(batch, tree); batch = []; size = 0
        batch.append(entry); size += amount
    if batch: tree = write_tree(batch, tree)
    assert tree == args.expected_tree, 'Remote tree differs from the immutable staged snapshot.'
    assert remote_head() == args.parent, 'Remote advanced during upload; no branch update performed.'
    commit = call('github_create_commit', dict(parent_sha=args.parent, tree_sha=tree,
        message=Path(args.message_file).read_text()))['sha']
    result = call('github_update_ref', dict(branch_name='main', sha=commit, force=False))
    assert result.get('success') is True
    assert remote_head() == commit
    receipt = dict(repository=args.repository, parent=args.parent, commit=commit, tree=tree,
                   files=len(entries) + len(text_entries), label=args.label)
    (state / (args.label + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    # Preserve ongoing changes in the working tree; never reset them to a checkpoint.
    assert git('write-tree').decode().strip() == args.expected_tree, 'Staged index changed; synchronize local refs manually.'
    subprocess.run(['git', 'fetch', 'origin'], cwd=root, check=True)
    assert git('rev-parse', 'origin/main').decode().strip() == commit
    assert git('rev-parse', 'origin/main^{tree}').decode().strip() == tree
    subprocess.run(['git', 'reset', '--mixed', 'origin/main'], cwd=root, check=True, stdout=subprocess.DEVNULL)
    print('PUBLISHED ' + json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
