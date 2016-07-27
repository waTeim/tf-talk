#!/usr/bin/env python

import sys
import os
import subprocess
import argparse

def job_epoch(cmd,args):
   env = os.environ.copy()
   env['CUDA_VISIBLE_DEVICES'] = "0"
   return subprocess.Popen(["/usr/bin/python",cmd] + args,stdout=subprocess.PIPE,env=env).communicate()[0].rstrip(os.linesep)

parser = argparse.ArgumentParser(description="run a NN script endlessly")
parser.add_argument('cmd')
parser.add_argument('cmdargs',nargs="*")

def main():
   i = 1
   args = parser.parse_args(sys.argv[1:])
   while True:
      print("******* iteration " + str(i) + " ********")
      output = job_epoch(args.cmd,args.cmdargs)
      print(output)
      i = i + 1

main()

