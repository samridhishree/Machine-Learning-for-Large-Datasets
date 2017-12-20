#!/usr/bin/env python

import sys, os, time, argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
default_host_file = os.path.join(script_dir, 'hosts.txt')
  
parser = argparse.ArgumentParser(description='Syncs the JBosen LDA application.')
parser.add_argument('--host_file', type=str, default=default_host_file, help='Path to the host file to use.')
parser.add_argument('--pem_file', type=str, default='',help='Location of AWS pem file')
args = parser.parse_args()


sync_path = "~/lda_py"

with open(args.host_file, "r") as f:
  ips = f.read().splitlines()

for ip in ips:
  ip = ip.split(":")[0]
  if args.pem_file:
    aws_args = "-i " + args.pem_file
  else:
    aws_args = " "
 

  ssh_cmd = "ssh -o StrictHostKeyChecking=no " + aws_args
  cmd = "rsync -avhce \" " + ssh_cmd + " \" " \
      + sync_path + " " + ip.strip() + ":~/"
  print cmd
  time.sleep(2)
  os.system(cmd)
