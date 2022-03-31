import sys
import pickle
from krips_alpha import krippendorff_alpha, nominal_metric


def get_model_labels(model_output):
	model_labels = []
	for x in model_output:
		try:
			if x['pos_score'] > x['neg_score']:
				model_labels.append('0')
			elif x['neg_score'] > x['pos_score']:
				model_labels.append('1')
			else:
				print(x, "error")
		except KeyError:
			if x['pos'] > x['neg']:
				model_labels.append('0')
			elif x['neg'] > x['pos']:
				model_labels.append('1')
			else:
				print(x, "error")
	return model_labels

annotations = pickle.load(open(sys.argv[1], 'rb'))
#print(len(annotations[0]))

alpha = krippendorff_alpha(annotations, nominal_metric)
#print(alpha)

model_output = pickle.load(open(sys.argv[2], 'rb'))
print(len(model_output))

model_labels = get_model_labels(model_output)
annotations.append(model_labels)
print(len(annotations))
model_alpha = krippendorff_alpha(annotations, nominal_metric)
print(model_alpha)
