import os


def remove_file(file):
    if os.path.exists(f"{file}.py"):
        # if windows
        if os.name == "nt":
            os.system(f"DEL {file}.py")
        # if linux
        else:
            os.system(f"rm {file}.py")


if __name__ == "__main__":
    file = "gumbell_sinkhorn_networks_experiments"
    try:
        remove_file(file)

        command = f"jupyter nbconvert --to script {file}.ipynb"
        os.system(command)  # convert the notebook to a python script

        # run the script
        print(f"Running {file}")
        exec(f"import {file}")

        # os.system(f"python -m {file}")
    except Exception as e:
        print(e)
        print("Could not run the script")
    finally:
        remove_file(file)

    print("Done")
