"""Named RSS arrangement with larger storage, unchanged contact law and rates.

This is environment configuration only. No capacity/readback value is exposed
to a policy. The stock arrangement remains registered under its original name.
"""


def arrangement_name(capacity=None):
    if capacity is None:
        return 'mujoco_500hz@v1'
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 512:
        raise ValueError('RSS contact capacity must be an integer >= 512')
    import robosimstudio as rss
    import robosimstudio_bench as bench

    baseline = rss.MUJOCO_500HZ
    old_backend = baseline.backend_for('*')
    if old_backend is None or old_backend.solver != 'mujoco':
        raise ValueError('Expected the stock MuJoCo 500 Hz arrangement')
    arrangement = baseline.with_backend(
        '*', old_backend.with_(nconmax=capacity, njmax=capacity))
    name = f'puzzler_mujoco_500hz_capacity{capacity}@v1'
    bench.register_arrangement(name, arrangement)
    return name
