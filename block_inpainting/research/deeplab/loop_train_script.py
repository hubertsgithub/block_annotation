from argparse import ArgumentParser
import shutil
import os
import subprocess

parser = ArgumentParser()
parser.add_argument('script', help='training script to run')
parser.add_argument('--num_iterations', type=int,
                    default=100000, help='number of iterations to run')
parser.add_argument('--start_iteration', type=int,
                    default=0, help='iteration to start on')
parser.add_argument('--eval_interval', type=int,
                    default=10000, help='evaluate every X iterations')
args = parser.parse_args()

args.script = os.path.abspath(args.script)


num_it = args.num_iterations
start_it = args.start_iteration
eval_int = args.eval_interval

first_stop_iteration = (start_it // eval_int + 1) * eval_int

for stop_it in range(first_stop_iteration, num_it + 1, eval_int):
    print('#######################################')
    print( 'Setting NUM_ITERATIONS TO:        {}'.format(num_it))
    print( '   Validation interval is:        {}'.format(eval_int))
    print( '   Next validation interation is: {}'.format(stop_it))
    print('#######################################')
    call = ['bash', args.script, str(num_it), str(eval_int), str(stop_it)]
    print('Calling script ...'.format(call))
    subprocess.call(call)
