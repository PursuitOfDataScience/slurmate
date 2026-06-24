# slurmate demo — render with:  vhs assets/demo.tape   (produces assets/demo.gif)
# Requires https://github.com/charmbracelet/vhs  (conda: vhs ttyd ffmpeg)
#
# Drives the REAL slurmate TUI. We strip Slurm from PATH so slurmate falls back
# to its built-in mock cluster (safe: no real sinfo/sbatch), while a throwaway
# .slurmate.toml (the base64 below — same as assets/demo.slurmate.toml) prefills
# the wizard with a GPU job. The tape types the job name + partition and walks
# the wizard one Enter at a time while the live #SBATCH preview grows, then
# submits on the Review step (mock cluster, so nothing is actually queued).

Output assets/demo.gif

Set Shell bash
Set FontSize 17
Set Width 1300
Set Height 820
Set Padding 18
Set Framerate 24
Set Theme "Catppuccin Mocha"

Hide
Type `export PATH=$(python -c 'import os;print(":".join(p for p in os.environ["PATH"].split(":") if "slurm" not in p.lower()))')`
Enter
Type `mkdir -p /tmp/sm-demo/logs && cd /tmp/sm-demo && unset SLURMATE_MOCK`
Enter
Type "echo YWNjb3VudCA9ICJteV9sYWIiCnBhcnRpdGlvbiA9ICJncHUtaGlnaGVuZCIKY3B1cyA9IDgKbWVtb3J5ID0gIjY0RyIKdGltZV9saW1pdCA9ICIwODowMDowMCIKbm9kZXMgPSAxCmdwdXMgPSAxCmdwdV90eXBlID0gImgxMDAiCmdwdV9mb3JtYXQgPSAiZ3Jlc190eXBlIgplbnZfdHlwZSA9ICJjb25kYSIKZW52X25hbWUgPSAicHl0b3JjaCIKbW9kdWxlcyA9IFsiY3VkYS8xMi4xIl0Kb3V0cHV0X2RpciA9ICJsb2dzIgpjb21tYW5kID0gInB5dGhvbiB0cmFpbi5weSAtLWVwb2NocyA1MCAtLWFtcCIK | base64 -d > .slurmate.toml && clear"
Enter
Show

Type "slurmate"
Enter
Sleep 2500ms

# Step 1 — Job name
Type "diffusion-train"
Sleep 800ms
Enter
Sleep 800ms

# Step 2 — Partition: pick "enter manually", then type the name
Enter
Sleep 600ms
Type "gpu-highend"
Sleep 700ms
Enter
Sleep 900ms

# Steps 3–20 — walk the config‑prefilled answers one Enter at a time,
# pausing on each so the live #SBATCH preview stays readable as it grows.
# Paced individually (NOT a bulk Enter@…repeat, which VHS buffers and
# flashes the GPU steps past in a single frame).
Enter
Sleep 550ms   # Account
Enter
Sleep 550ms   # QoS
Enter
Sleep 550ms   # CPU cores
Enter
Sleep 550ms   # Memory
Enter
Sleep 550ms   # Time limit
Enter
Sleep 550ms   # Nodes
Enter
Sleep 550ms   # Tasks per node
Enter
Sleep 900ms   # GPUs
Enter
Sleep 900ms   # GPU type
Enter
Sleep 900ms   # GPU format
Enter
Sleep 600ms   # Array spec
Enter
Sleep 650ms   # Output directory
Enter
Sleep 650ms   # Output file
Enter
Sleep 650ms   # Custom #SBATCH flags
Enter
Sleep 800ms   # Modules
Enter
Sleep 1000ms  # Environment type
Enter
Sleep 800ms   # Environment name/path
Enter
Sleep 1000ms  # Command to run

# Step 21 — Review & Submit: hold on the full script, then submit
# (mock cluster, so nothing is actually queued).
Sleep 2500ms
Enter
Sleep 3s
