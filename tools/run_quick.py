"""Execute a DPC notebook headless in QUICK mode (Task 0.4 smoke test / regression), CPU.
Usage: python tools/run_quick.py <notebook.ipynb> <work_dir> [timeout_s]
The notebook's %%writefile cells write the modules into <work_dir>; results go to <work_dir>/dpc_out_quick (DPC_OUT_ROOT).
Only the settings cell is changed: QUICK = True. Executed notebook saved as <work_dir>/executed.ipynb."""
import os, sys, time
import nbformat
from nbclient import NotebookClient
nb_path, work = sys.argv[1], os.path.abspath(sys.argv[2])
os.makedirs(work, exist_ok=True)
nb = nbformat.read(nb_path, 4)
n = 0
for c in nb.cells:
    if c.cell_type == "code" and c.source.startswith("QUICK  = False"):
        c.source = c.source.replace("QUICK  = False", "QUICK  = True", 1); n += 1
assert n == 1, "settings cell not found"
os.environ["DPC_OUT_ROOT"] = os.path.join(work, "dpc_out")
t0 = time.time()
client = NotebookClient(nb, timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 7200, kernel_name="lk",
                        resources={"metadata": {"path": work}}, allow_errors=False)
try:
    client.execute()
    status = "OK"
except Exception as e:  # noqa
    status = f"ERROR: {type(e).__name__}: {str(e)[-2000:]}"
nbformat.write(nb, os.path.join(work, "executed.ipynb"))
print(f"{nb_path}: {status} | {time.time() - t0:.0f}s")
