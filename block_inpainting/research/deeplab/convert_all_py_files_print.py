import glob
import shutil
import os

all_py_files = glob.glob("./**/*.py", recursive=True)
print(all_py_files)

for py_file in all_py_files:
    if not os.path.exists(py_file+".backup"):
        shutil.copy(src=py_file, dst=py_file+".backup")
    lines = []
    print ("###### WORKING ON: {} ########".format(py_file))
    for line in open(py_file):
        if 'print ' in line and 'print (' not in line:
            print(line)
            line = line.replace("print(", "print(").replace("\n", ")\n"))
            print(line)
            print("===")
        lines.append(line)
    print("".join(lines))
    fp = open(py_file, "w+")
    fp.write("".join(lines))

