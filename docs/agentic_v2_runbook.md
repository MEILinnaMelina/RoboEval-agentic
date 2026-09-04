# Agentic v2 Runbook

Use the environment Python directly in PowerShell:

     = "C:\Users\melin\.conda\envs\roboeval\python.exe"

Run one fixed-plan trial with a visible window and saved GIF:

    &  examples\run_agentic_v2.py --task cube_handover --seed 0 --planner fixed --render window --record-gif

Run one real online OpenAI trial:

    &  examples\run_agentic_v2.py --task cube_handover --seed 0 --planner openai --model gpt-5.6-terra --render window --record-gif

The report is saved under
outputs\agentic_v2\METHOD\TASK\seed_000\trial_report.json. The same folder
contains separate planner and deterministic planning traces.

Create the Phase 11 manifest and aggregate existing reports without launching
experiments:

    &  examples\evaluate_agentic_v2.py

Launch only the deterministic 30-trial gate:

    &  examples\evaluate_agentic_v2.py --launch --methods v2-fixed

Launch selected API rows only after fixed plans pass at least 8/10 for every
task:

    &  examples\evaluate_agentic_v2.py --launch --methods v2-full-no-replan v2-full

Historical rows run from read-only worktrees pinned to their recorded commits.
Pass those roots when launching the full seven-row matrix:

    &  examples\evaluate_agentic_v2.py --launch --methods v1-p22-independent v1-p23-memory v2-ik-only v2-fixed v2-full-no-replan v2-full v2-full-memory --v1-p22-root E:\djf\RoboEval-v1-p22 --v1-p23-root E:\djf\RoboEval-v1-p23

To record model cost, set current per-million-token rates before launching:

     = "<current input rate>"
     = "<current output rate>"

The launcher creates a fresh process and environment for every method/task/seed
trial. The memory row passes only an explicit per-task memory file; all other
rows start without cross-trial memory.
