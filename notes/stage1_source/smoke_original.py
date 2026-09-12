"""Short full-graph Brian2 run; no training and no behavioural claim."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from brian2 import Network, prefs, seed, ms, mV

ROOT = Path(__file__).resolve().parent


def main():
    prefs.codegen.target = 'numpy'
    seed(123)
    spec = importlib.util.spec_from_file_location('upstream_fly', ROOT / 'upstream/model.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    params = dict(upstream.default_params)
    # Upstream resets an undeclared variable `w`; remove only that assignment in
    # this wrapper. Keep the downloaded file byte-identical to the pinned source.
    old_reset = params['eq_rst']
    params['eq_rst'] = 'v = v_rst; g = 0 * mV'
    t0 = time.perf_counter()
    neurons, synapses, monitor = upstream.create_model(
        str(ROOT / 'data/raw/Completeness_783.csv'),
        str(ROOT / 'data/raw/Connectivity_783.parquet'), params)
    audit = json.loads((ROOT / 'results/data_audit.json').read_text())
    source_index = audit['pilot']['center_source_index']
    neurons.v[source_index] = -40 * mV
    net = Network(neurons, synapses, monitor)
    net.run(10 * ms)
    result = dict(status='passed', neurons=len(neurons), directed_connections=len(synapses),
                  biological_time_ms=10, injected_neuron_source_index=source_index,
                  total_spikes=int(monitor.num_spikes),
                  neurons_that_spiked=int(len(np.unique(np.asarray(monitor.i)))),
                  voltage_finite=bool(np.isfinite(np.asarray(neurons.v)).all()),
                  original_reset=old_reset, wrapper_reset=params['eq_rst'],
                  elapsed_seconds=time.perf_counter() - t0,
                  interpretation='Numerical smoke only. No training, language, body, or validation of behaviour.')
    assert result['voltage_finite'] and result['total_spikes'] > 0
    (ROOT / 'results/original_brian2_smoke.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
