"""Test LIBERO installation."""
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

bm = benchmark.get_benchmark_dict()
spatial_bm = bm["libero_spatial"]()
print("Tasks:", spatial_bm.get_task_names())
print("Num tasks:", len(spatial_bm.get_task_names()))

task = spatial_bm.get_task(0)
print("Task name:", task.name)
print("Task problem:", task.problem)
print("Task bddl_file:", task.bddl_file)

env = OffScreenRenderEnv(
    bddl_file_name=task.bddl_file,
    episode_length=100,
    camera_heights=128,
    camera_widths=128,
)
obs, info = env.reset()
print("Obs keys:", list(obs.keys()))
for k, v in obs.items():
    print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
env.close()
print("Success!")
