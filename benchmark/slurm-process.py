filepath = '/Users/scohen/Downloads/slurm-88827.out'

datasets = ['housing', # 'compactiv',
 'delta-elevators', 'redwine', 'delta-ailerons', 'whitewine', 'ailerons']

names = { 'housing': 'Housing', 'delta-elevators':'DeltaElevators', 'delta-ailerons':'DeltaAilerons', 'ailerons':'Ailerons', 'redwine':'RedWine', 'whitewine':'WhiteWine' }

results = {}

for dataset in datasets:
	results[dataset] = {}
	for k in range(5):
		results[dataset][k] = {}
		results[dataset][k]['qp_err'] = 0
		results[dataset][k]['qp_init_err'] = 0
		results[dataset][k]['qp_init_epochs'] = 0
		results[dataset][k]['bp_err'] = []
		results[dataset][k]['bp_epochs'] = []

with open(filepath) as fp:
	line = fp.readline()
	cnt = 1
	while line:
		while ('  ' in line):
			line = line.replace('  ', ' ')

		words = line.split(' ');
		if line.startswith('['):
			dataset = words[0].replace('[', '').replace(']','')

			if dataset in results:
				if 'fold' in line:
					fold = int(words[2].replace(']',''))-1
				if ('qp_err' in line):
					qp_err = float(words[7])
					results[dataset][fold]['qp_err'] = qp_err

				if ('bp_err' in line) and ('myinit' in line):
					bp_err = float(words[15])
					bp_epochs = int(words[4])
					if ('myinit: 0' in line):
						results[dataset][fold]['qp_init_err'] = bp_err
						results[dataset][fold]['qp_init_epochs'] = bp_epochs

					else:
						results[dataset][fold]['bp_err'].append(bp_err)					
						results[dataset][fold]['bp_epochs'].append(bp_epochs)



		line = fp.readline()
		cnt += 1
		
		
cnt = 0
for dataset in names:
	r = results[dataset]
	
	qp = 0
	bp = 0
	qp_init = 0
	qp_init_e = 0
	bp_e = 0
	
	all_bp = []
	all_qp = []
	all_qp_init = []
	all_bp_folds = []

	cnt = 0
	
	for f in range(5):
		qp = qp + r[f]['qp_err']
		
		all_qp.append( r[f]['qp_err'])
		
		for [bp_err,bp_epochs] in zip(r[f]['bp_err'], r[f]['bp_epochs']):
			bp = bp + bp_err
			bp_e = bp_e + bp_epochs
			all_bp.append(bp_err)
			cnt = cnt + 1
	
		all_bp_folds.append(bp / cnt)
		
		qp_init = qp_init + r[f]['qp_init_err']
		qp_init_e = qp_init_e + r[f]['qp_init_epochs']
		all_qp_init.append( r[f]['qp_init_err'])
	
	
	bp = bp / cnt
	bp_e = bp_e / cnt
	qp = qp / 5
	qp_init_e = qp_init_e / 5
	qp_init = qp_init / 5
	
	if ('ailerons' in dataset or 'delta' in dataset):	
		print('\\textsc{', names[dataset], '}', ' & ', "{:.5f}".format(qp), ' & ', "{:.5f}".format(bp), '{\\small /',  int(bp_e),  '}', ' & ', "{:.5f}".format(qp_init), '{\\small /', int(qp_init_e),  '} \\\\')

	else:
		print('\\textsc{', names[dataset], '}', ' & ', "{:.2f}".format(qp), ' & ', "{:.2f}".format(bp), '{\\small /',  int(bp_e),  '}', ' & ', "{:.2f}".format(qp_init), '{\\small /', int(qp_init_e),  '} \\\\')



#		print(dataset, ' & ', "{:.2f}".format(qp), ' & ', "{:.2f}".format(bp), '{\\small \\pm ',  "{:.2f}".format(bp_e),  '}', ' & ', "{:.2f}".format(qp_init), '{\\small \\pm ',  "{:.2f}".format(qp_init_e),  '} \\\\')
