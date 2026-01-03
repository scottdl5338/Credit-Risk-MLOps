import subprocess
# subprocess - llows Python to start other programs (like another Python script) and wait for them to finish.
import os
import sys
# sys - Gives you access to variables used by the Python interpreter itself.

# This finds the exact path to the Python running this script (which is your .venv)
python_exe = sys.executable 

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    # Use the specific .venv python to run the sub-scripts
    result = subprocess.run([python_exe, script_name], capture_output=True, text=True)
    # .run() - Go do this task
    # [python_exe, script_name] - he equivalent of typing python preprocess.py in your terminal.
    # capture_output=True - Catches all print statements for terminal
    # text = True - he output as readable English (strings) instead of raw computer binary code

    # After every program if 0 returned 0 the sucess else 1 = error
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error in {script_name}:")
        print(result.stderr)
        # .stderr - Returns Traceback error 
        exit(1)
        # exit(1) - Stops entire pipeline

run_script('preprocess.py')
run_script('train_model.py')
run_script('predict.py')