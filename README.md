Setting virtual environment (via conda)

`conda create -n <name_of_venv> python=3.7.6`

`conda activate <name_of_venv>`

If using PyCharm, add this venv in Preferences > Project > Project Interpreter

Install all requirements from requirements.txt:

`pip install -r requirements.txt`

Install kernel in jupyter notebook representing the virtual environment created:

`python -m ipykernel install --user --name <kernel_name> --display-name "<Kernel display-name>"`
