Quick minimal steps to run RainyDay from the terminal (avoid complex IDE flows)

1) Inspect the environment name that the YAML may create:

   grep -E '^name:' RainyDay_Env.yml || head -n 10 RainyDay_Env.yml

2) Create the conda environment (use mamba if available):

   conda env create -f RainyDay_Env.yml
   # or with explicit name
   conda env create -f RainyDay_Env.yml -n RainyDay_Env

3) Activate the environment in bash:

   conda activate RainyDay_Env

   # If 'conda' isn't available in your shell session, do:
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate RainyDay_Env

4) Run RainyDay using the helper script (made to keep things simple):

   # make script executable once
   chmod +x scripts/run_rainyday.sh

   # run (ENV_NAME optional, defaults to 'RainyDay_Env')
   ./scripts/run_rainyday.sh RainyDay_Env Examples/BigThompson/BigThompsonExample.json

Notes & Troubleshooting
- If environment creation fails with package conflicts, try installing 'mamba' and re-run with:
  mamba env create -f RainyDay_Env.yml

- If RainyDay needs a parameter JSON file, use one of the example JSONs under `Examples/` (e.g. `Examples/BigThompson/BigThompsonExample.json`).

- To test quickly without real data, you can create a minimal params JSON that points at tiny test inputs. If you'd like, I can create a minimal test parameter file next.

- To avoid the IDE entirely: open Terminal (bash) and run the commands above. The helper script will activate the conda env and run the main script for you.

Want me to also create a tiny minimal parameter JSON (a safe stub) and run a local quick smoke test? If yes, tell me which Example you want used as a base (BigThompson or Madison) or say 'create stub'.
