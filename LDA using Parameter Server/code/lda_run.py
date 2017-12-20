#!/usr/bin/env python

if __name__ == '__main__':
  
  import argparse
  import os
  import time

  script_dir = os.path.dirname(os.path.realpath(__file__))
  default_host_file = os.path.join(script_dir, 'hosts.txt')
  default_data_file = os.path.join(script_dir, '20news.csv')
  
  parser = argparse.ArgumentParser(description='Launches the JBosen LDA application using SSH.')
  parser.add_argument('--host_file', type=str, default=default_host_file, help='Path to the host file to use.')
  parser.add_argument('--num_local_worker_threads', type=int, default=1, help='Number of application worker threads per client.')
  parser.add_argument('--num_local_comm_channels', type=int, default=1, help='Number of network channels per client.')
  parser.add_argument('--data_file', type=str, default=default_data_file, help='Path to the input data file to use.')
  parser.add_argument('--output_dir', type=str, default=script_dir, help='Path to the output directory to use.')
  parser.add_argument('--num_words', type=int, default=None, help='Number of words in the vocabulary.')
  parser.add_argument('--num_topics', type=int, default=1, help='Number of topics to run LDA with.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Value of alpha.')
  parser.add_argument('--beta', type=float, default=1.0, help='Value of beta.')
  parser.add_argument('--num_iterations', type=int, default=1, help='Number of passes over the dataset to run.')
  parser.add_argument('--num_clocks_per_iteration', type=int, default=1, help='Number of clocks for each iteration.')
  parser.add_argument('--staleness', type=int, default=0, help='Number of clocks for each iteration.')
  parser.add_argument('--java_args', type=str, default='', help='Extra arguments to pass to Java.')
  parser.add_argument('--pem_file', type=str, default='',help='Location of AWS pem file')
  args = parser.parse_args()
  
  #class_path = os.path.join(script_dir, 'build', 'libs', 'Lda.jar')
  main_class = 'Lda'

  with open(args.host_file, 'r') as f:
    host_ips = [line.split(':')[0] for line in f]

  def launch(client_id, ip):
    if args.pem_file:
        aws_args = "-i " + args.pem_file
    else:
        aws_args = " "
    cmd = 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ' + aws_args + " " +  ip + ' '
    cmd += '"cd ' + os.getcwd() + '; '
    cmd += 'jython Lda.py '
    cmd += '-clientId %d ' % client_id
    cmd += '-hostFile %s ' % args.host_file
    cmd += '-numLocalWorkerThreads %d ' % args.num_local_worker_threads
    cmd += '-numLocalCommChannels %d ' % args.num_local_comm_channels
    cmd += '-dataFile %s ' % args.data_file
    cmd += '-outputDir %s ' % args.output_dir
    cmd += '-numWords %d ' % args.num_words
    cmd += '-numTopics %d ' % args.num_topics
    cmd += '-alpha %f ' % args.alpha
    cmd += '-beta %f ' % args.beta
    cmd += '-numIterations %d ' % args.num_iterations
    cmd += '-numClocksPerIteration %d ' % args.num_clocks_per_iteration
    cmd += '-staleness %d ' % args.staleness
    cmd += '" &'
    print(cmd)
    os.system(cmd)

  if args.num_words is None:
    print 'Counting number of words in \'%s\'...' % args.data_file
    print 'To avoid this, provide the \'--num_words\' argument.'
    max_word = -1
    with open(args.data_file, 'r') as f:
      for line in f:
        words = map(int, line.strip().split(','))
        max_word = max(max_word, max(words))
    args.num_words = max_word + 1

  print("Starting instances of LDA...")
  for client_id, ip in enumerate(host_ips):
    launch(client_id, ip)
