import sys
sys.path.append("Lda.jar")

from org.petuum.jbosen import PsApplication, PsTableGroup, PsConfig
from org.kohsuke.args4j import Option

import argparse
from LdaApp import LdaApp


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-clientId', dest='clientId', type=int, required=True, help="Client ID")
	parser.add_argument('-hostFile', dest='hostFile', type=str, required=True, help="Path to host file")
	parser.add_argument('-numLocalWorkerThreads', dest='numLocalWorkerThreads', type=int, required=True, help='Number of application worker threads per client')
	parser.add_argument('-numLocalCommChannels', dest='numLocalCommChannels', type=int, required=True, help='Number of network channels per client')
	parser.add_argument('-dataFile', dest='dataFile', type=str, required=True, help="Path to data file.")
	parser.add_argument('-outputDir', dest='outputDir', type=str, required=True, help="Path to output dir.")
	parser.add_argument('-numWords', dest='numWords', type=int, required=True, help="Number of words in the vocabulary.")
	parser.add_argument('-numTopics', dest='numTopics', type=int, required=True, help="Number of topics.")
	parser.add_argument('-alpha', dest='alpha', type=float, required=True, help="Alpha.")
	parser.add_argument('-beta', dest='beta', type=float, required=True, help="Beta.")
	parser.add_argument('-numIterations', dest='numIterations', type=int, required=True, help="Number of iterations.")
	parser.add_argument('-numClocksPerIteration', dest='numClocksPerIteration', type=int, required=True, help="Number of clocks for each iteration.")
	parser.add_argument('-staleness', dest='staleness', type=int, default=0, help="Path to data file.")
	params = vars(parser.parse_args())

	ldaApp = LdaApp(params['dataFile'], params['outputDir'], params['numWords'], params['numTopics'],
					params['alpha'], params['beta'], params['numIterations'], params['numClocksPerIteration'],
					params['staleness'])
	config = PsConfig()
	config.clientId = params['clientId']
	config.hostFile = params['hostFile']
	config.numLocalWorkerThreads = params['numLocalWorkerThreads']
	config.numLocalCommChannels= params['numLocalCommChannels']
	ldaApp.run(config)



if __name__ == "__main__":
	main()