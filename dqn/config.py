config = dict({
	"LunarLander-v2": {

		"DQN": {

			"batch_size" : 128,
			"eps_decay"  : 0.99,
			"gamma"      : 0.99,
			"tau"        : 0.005,
			"lr"         : 0.0005

		},

		"EnsembleDQN": {

			"batch_size" : 64,
			"eps_decay"  : 0.99,
			"gamma"      : 0.99,
			"tau"        : 0.005,
			"lr"         : 0.0005

		},

		"IV_DQN": {

			"batch_size"    : 64,
			"eps_decay"     : 0.99,
			"gamma"         : 0.99,
			"tau"           : 0.005,
			"lr"            : 0.0005,
			"minimal_eff_bs": None

		}


	}

})