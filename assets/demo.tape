# slurmate demo — render with:  vhs assets/demo.tape   (produces assets/demo.gif)
# Requires https://github.com/charmbracelet/vhs  (conda: vhs ttyd ffmpeg)
#
# Drives the REAL slurmate TUI. We strip Slurm from PATH so slurmate falls back
# to its built-in mock cluster (safe: no real sinfo/sbatch), while a throwaway
# .slurmate.toml (the base64 below — same as assets/demo.slurmate.toml) prefills
# the wizard with a GPU job. The tape types the job name + partition and walks
# the wizard with Enter while the live #SBATCH preview grows, ending on Review.

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
Sleep 3s

Type "diffusion-train"
Sleep 600ms
Enter
Sleep 900ms
Enter
Sleep 700ms
Type "gpu-highend"
Sleep 500ms
Enter
Sleep 900ms
Enter@520ms 18

Sleep 4s
Enter
Sleep 3s
