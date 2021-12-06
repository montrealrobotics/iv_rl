config = dict({
	"LunarLander-v2": {

		"DQN": {

			"eff_batch_size" : 128,
			"eps_decay"  : 0.99,
			"gamma"      : 0.99,
			"tau"        : 0.005,
			"lr"         : 0.0005

		},

		"EnsembleDQN": {

			"eff_batch_size" : 64,
			"eps_decay"  : 0.99,
			"gamma"      : 0.99,
			"tau"        : 0.005,
			"lr"         : 0.0005

		},

		"BootstrapDQN":{

			"eff_batch_size"     : 64,
			"eps_decay"          : 0.99,
			"gamma"              : 0.99,
			"tau"                : 0.005,
			"lr"                 : 0.0005,
			"mask"               : "bernoulli",
			"mask_prob"          : 0.9,
			"prior_scale"        : 10

		},

		"ProbDQN":{

			"eff_batch_size"     : 256,
			"eps_decay"          : 0.991,
			"gamma"              : 0.99,
			"tau"                : 0.001,
			"lr"                 : 0.0005,
			"loss_att_weight"    : 2
		},

		"IV_EnsembleDQN": {

			"eff_batch_size"     : 64,
			"eps_decay"          : 0.99,
			"gamma"              : 0.99,
			"tau"                : 0.005,
			"lr"                 : 0.0005,
			"dynamic_eps"        : True,
			"minimal_eff_bs"     : 48,
	
		},

		"IV_BootstrapDQN":{

			"eff_batch_size"     : 64,
			"eps_decay"          : 0.99,
			"gamma"              : 0.99,
			"tau"                : 0.005,
			"lr"                 : 0.0005,
			"dynamic_eps"        : True,
			"mask"               : "bernoulli",
			"mask_prob"          : 0.5,
			"minimal_eff_bs"     : 48,
			"prior_scale"        : 0.1
		},

		"IV_ProbEnsembleDQN":{

			"eff_batch_size"     : 64,
			"eps_decay"          : 0.99,
			"gamma"              : 0.99,
			"tau"                : 0.005,
			"lr"                 : 0.001,
			"eps"                : 10,
			"loss_att_weight"    : 3
		},


		"IV_ProbDQN":{

			"eff_batch_size"    : 256,
			"eps_decay"         : 0.991,
			"gamma"             : 0.99,
			"tau"               : 0.001,
			"lr"                : 0.0005,
			"loss_att_weight"   : 2,
			"dynamic_eps"       : True,
			"minimal_eff_bs"    : 208
		}

	},

	"MountainCar-v0":{

		"DQN":{
			"eff_batch_size"    : 256,
			"lr"                : 0.001,
			"eps_decay"         : 0.98,
			"tau"               : 0.01
		},

		"BootstrapDQN":{
			"eff_batch_size"    : 256,
			"lr"                : 0.001,
			"eps_decay"         : 0.98,
			"tau"               : 0.05,
			"mask_prob"         : 0.5,
			"prior_scale"       : 10
		},

		"SunriseDQN":{
			"eff_batch_size"    : 256,
			"lr"                : 0.001,
			"eps_decay"         : 0.98,
			"tau"               : 0.05,
			"mask_prob"         : 0.5,
			"prior_scale"       : 10,
			"sunrise_temp"      : 50
		},

		"IV_DQN":{
			"eff_batch_size"    : 256,
			"lr"                : 0.001,
			"eps_decay"         : 0.98,
			"tau"               : 0.05,
			"mask_prob"         : 0.5,
			"prior_scale"       : 10,
			"eps"               : 1000
		},

		"IV_ProbEnsembleDQN":{
			"eff_batch_size"    : 256,
			"lr"                : 0.001,
			"eps_decay"         : 0.98,
			"tau"               : 0.05,
			"mask_prob"         : 0.5,
			"prior_scale"       : 10,
			"eps"               : 1000
		},


	},

	"gym_cheetah":{

		"EnsembleSAC":{

			"eff_batch_size"   : 1024,
			"mask_prob"        : 0.9,
			"ucb_lambda"       : 0

		},

		"IV_EnsembleSAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 0.9,
			"ucb_lambda"            : 10,
			"minimal_eff_bs_ratio"  : 0.99,
			"dynamic_eps"           : True 

		},


		"IV_ProbEnsembleSAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 1,
			"ucb_lambda"            : 0,
			"minimal_eff_bs_ratio"  : 0.99,
			"dynamic_eps"           : True,
			"loss_att_weight"       : 2


		},

		"IV_SAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 1,
			"ucb_lambda"            : 0,
			"minimal_eff_bs_ratio"  : 0.99,
			"dynamic_eps"           : True,
			"loss_att_weight"       : 2


		},

		"IV_ProbSAC":{
			"loss_att_weight"       : 5,
			"minimal_eff_bs_ratio"  : 0.5
		}

	},


	"gym_walker2d":{

		"EnsembleSAC":{

			"eff_batch_size"   : 512,
			"mask_prob"        : 1,
			"ucb_lambda"       : 1

		},


		"IV_EnsembleSAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 0.9,
			"ucb_lambda"            : 10,
			"minimal_eff_bs_ratio"  : 0.8,
			"dynamic_eps"           : True 

		},


		"IV_ProbEnsembleSAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 0.9,
			"ucb_lambda"            : 10,
			"minimal_eff_bs_ratio"  : 0.8,
			"dynamic_eps"           : True,
			"loss_att_weight"       : 5

		},

		"IV_SAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 0.9,
			"ucb_lambda"            : 10,
			"minimal_eff_bs_ratio"  : 0.8,
			"dynamic_eps"           : True,
			"loss_att_weight"       : 5

		},


	},


	"gym_hopper":{

		"EnsembleSAC":{

			"eff_batch_size"   : 512,
			"mask_prob"        : 1,
			"ucb_lambda"       : 10
		},


		"IV_ProbEnsembleSAC":{

				"eff_batch_size"        : 1024,
				"mask_prob"             : 0.7,
				"ucb_lambda"            : 10,
				"minimal_eff_bs_ratio"  : 0.8,
				"dynamic_eps"           : True,
				"loss_att_weight"       : 10

		},

		"IV_SAC":{

				"eff_batch_size"        : 1024,
				"mask_prob"             : 0.7,
				"ucb_lambda"            : 10,
				"minimal_eff_bs_ratio"  : 0.8,
				"dynamic_eps"           : True,
				"loss_att_weight"       : 10

		},


	},


	"gym_ant":{

		"EnsembleSAC":{

			"eff_batch_size"   : 512,
			"mask_prob"        : 0.9,
			"ucb_lambda"       : 10

		},


		"IV_ProbEnsembleSAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 1,
			"ucb_lambda"            : 1,
			"minimal_eff_bs_ratio"  : 0.9,
			"dynamic_eps"           : True,
			"loss_att_weight"       : 5

		},

		"IV_SAC":{

			"eff_batch_size"        : 1024,
			"mask_prob"             : 1,
			"ucb_lambda"            : 1,
			"minimal_eff_bs_ratio"  : 0.9,
			"dynamic_eps"           : True,
			"loss_att_weight"       : 5

		},
	},

	"cartpole":{

		"BootstrapDQN":{
			"batch_size"            : 128,
			"mask_prob"             : 5
		},

		"IV_BootstrapDQN":{
			"batch_size"            : 128,
			"mask_prob"             : 0.5,
			"minimal_eff_bs_ratio"  : 0.99
		},


		"IV_ProbEnsembleDQN":{
			"batch_size"            : 128,
			"mask_prob"             : 0.5,
			"minimal_eff_bs_ratio"  : 0.99,
			"loss_att_weight"       : 10
		},

		"IV_BootstrapDQN":{
			"batch_size"            : 128,
			"mask_prob"             : 0.5,
			"minimal_eff_bs_ratio"  : 0.99,
		},

		"IV_ProbDQN": {

			"loss_att_weight"       : 0.1,
			"minimal_eff_bs_ratio"  : 0.7
		},

		"ProbEnsembleDQN":{
			"batch_size"            : 128,
			"loss_att_weight"       : 10,
			"mask_prob"             : 0.5
		}
	}

	
})

