import sys

class DataLoader:
	def __init__(self, dataFile):
		self.dataFile = dataFile
  
	def load(self, part, numParts):
		lines = []
		try:
			with open(self.dataFile, 'r') as fr:
				for i, line in enumerate(fr):
					if i % numParts == part:
						lines.append(line)
		except Exception as detail:
			print(detail)
			sys.exit(1)
    
		w = []
		for i in range(len(lines)):
			tokens = lines[i].split(",")
			w.append([0] * len(tokens)) 
			for j in range(len(tokens)):
				w[i][j] = int(tokens[j])
    
		return w
