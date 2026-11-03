"""Execute a DPC notebook headless in QUICK mode (Task 0.4 smoke test / regression), CPU.
Usage: python tools/run_quick.py <notebook.ipynb> <work_dir> [timeout_s] [--set 'NAME = value' ...]
--set replaces the line starting with "NAME " (first occurrence in any code cell), e.g. --set 'ARCH   = "mlp"'.
The notebook's %%writefile cells write the modules into <work_dir>; results go to <work_dir>/dpc_out_quick (DPC_OUT_ROOT).
Only the settings cell is changed: QUICK = True. Executed notebook saved as <work_dir>/executed.ipynb."""
import os, sys, time
import nbformat
from nbclient import NotebookClient
args = [a for a in sys.argv[1:]]
sets = [args[i + 1] for i, a in enumerate(args) if a == "--set"]
pos = [a for i, a in enumerate(args) if a != "--set" and (i == 0 or args[i - 1] != "--set")]
nb_path, work = pos[0], os.path.abspath(pos[1])
os.makedirs(work, exist_ok=True)
nb = nbformat.read(nb_path, 4)
n = 0
for c in nb.cells:
    if c.cell_type == "code" and c.source.startswith("QUICK  = False"):
        c.source = c.source.replace("QUICK  = False", "QUICK  = True", 1); n += 1
assert n == 1, "settings cell not found"
for st in sets:
    name = st.split("=")[0].strip()
    done = False
    for c in nb.cells:
        if c.cell_type != "code" or done or c.source.startswith("%%writefile"):   # never edit the embedded modules
            continue
        lines = c.source.split("\n")
        for k, ln in enumerate(lines):
            if ln.split("=")[0].strip() == name and not ln.startswith(" "):
                lines[k] = st + "   # set by run_quick"; done = True; break
        c.source = "\n".join(lines)
    assert done, f"setting {name} not found"
os.environ["DPC_OUT_ROOT"] = os.path.join(work, "dpc_out")
t0 = time.time()
client = NotebookClient(nb, timeout=int(pos[2]) if len(pos) > 2 else 7200, kernel_name="lk",
                        resources={"metadata": {"path": work}}, allow_errors=False)
try:
    client.execute()
    status = "OK"
except Exception as e:  # noqa
    status = f"ERROR: {type(e).__name__}: {str(e)[-2000:]}"
nbformat.write(nb, os.path.join(work, "executed.ipynb"))
print(f"{nb_path}: {status} | {time.time() - t0:.0f}s")
