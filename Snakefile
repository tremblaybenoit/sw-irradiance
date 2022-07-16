#########################################################################################################
##SETUP AND GENERATE TRAINING DATA
#########################################################################################################
configfile: "snakemake-config.yaml"


## finds corresponding eve data given path to aia files
## make sure the output path matches your user home folder

rule create_eve_netcdf:
    input:
        eve_raw_path=config["eve_base_path"]+"/raw"
    output:
        eve_netcdf_path= config["sw-irr-output_path"]+"/EVE_irradiance.nc"
    shell:
        "python fdleuvai/data/preprocess/create_eve_netcdf.py -eve_raw_path {input.eve_raw_path} -netcdf_outpath {output.eve_netcdf_path}"


## generates matches between EVE data and the AIA files
rule generate_matches_time:
    input:
        eve_netcdf_path = config["sw-irr-output_path"]+"/EVE_irradiance.nc",
        aia_path = config["aia_path"]
    params:
        match_outpath = config["sw-irr-output_path"]
    output:
        matches_output = config["sw-irr-output_path"]+"/matches_eve_aia_171_193_211_304.csv"
    shell:
        "python fdleuvai/data/preprocess/generate_matches_time.py -eve_path {input.eve_netcdf_path} -aia_path {input.aia_path} -out_path {params.match_outpath}"

## generates train, test, values datasets 
rule make_train_val_test_sets:
    input:
        matches = config["sw-irr-output_path"]+"/matches_eve_aia_171_193_211_304.csv"
    output:
        expand(config["sw-irr-output_path"]+"/{split}.csv",split = config["SPLIT"])
    shell:
        "python fdleuvai/data/preprocess/make_train_val_test_sets.py --src {input.matches} --splits rve"
        

#########################################################################################################
##TEST AND TRAIN MODEL
#########################################################################################################

## fits means and stds to linear model
rule fit_linear_model:
    params:
        basepath = config["sw-irr-output_path"]
    input:
      expand(config["sw-irr-output_path"]+"/{split}.csv",split = config["SPLIT"]),
      eve_netcdf_path=config["eve_base_path"]+"/EVE_irradiance.nc"
    output:
        linear_preds =expand(config["sw-irr-output_path"]+"/EVE_linear_pred__{split}.nc",split = config["SPLIT"]),
        linear_stats =config["sw-irr-output_path"]+"/mean_std_feats.npz",
        eve_resid =config["sw-irr-output_path"]+"/eve_residual_{}_14ptot.npy"

    shell:
        """
        python canonical_code/fdl18_fit_linear_model.py \
        --base {params.basepath}
        """

## Creates the normalization values based on the train set
rule calculate_training_normalization:
    input:
        matches = config["sw-irr-output_path"]+"/matches_eve_aia_171_193_211_304.csv",
    params:
        basepath = config["sw-irr-output_path"],
        divide = 4
    output:
        norm_stats=expand(config["sw-irr-output_path"]+"/_{instrument}_{norm_stat}.npy",instrument=config["INSTRUMENT"],norm_stat=config["NORM-STATISTIC"])
    shell:
        """
        python fdleuvai/data/preprocess/calculate_training_normalization.py \
        --base {params.basepath} \
        --divide {params.divide}
        """

## train CNN
rule train_CNN:
    input:
        # means="eve_residual_mean_14ptot.npy",
        # stds="eve_residual_std_14ptot.npy",
        norm_stats=expand("{path}/_{instrument}_{norm_stat}.npy",path=config["sw-irr-output_path"],instrument=config["INSTRUMENT"],norm_stat=config["NORM-STATISTIC"]),
        linear_preds =expand(config["sw-irr-output_path"]+"/EVE_linear_pred__{split}.nc",split = config["SPLIT"]),
        linear_stats =config["sw-irr-output_path"]+"/mean_std_feats.npz",
    output:
        model = expand(config["sw-irr-output_path"]+"/EVE_linear_pred__{split}_model.pt",split = config["SPLIT"]),
        log = expand(config["sw-irr-output_path"]+"/EVE_linear_pred__{split}_log.txt",split = config["SPLIT"]),
        trained_loss = config["sw-irr-output_path"]+"trained_loss.npy",
        val_loss = config["sw-irr-output_path"]+"val_loss.npy"
    params:
        data_path = config["sw-irr-output_path"],
        model_results = config["model_results_path"]
    shell:
        """
        python fdl18_cdfg_residual_unified_train_to_tirr.py \
        --src {params.data_path} \
        --data_root {params.data_path} \
        --target {params.model_results} #model results folder
         """

## test data
rule test_CNN:
    input:
        trained_loss = "trained_loss.npy",
        val_loss = "val_loss.npy"
    output:
        errors = "errors.npy"
    params:
        phase = config["phase"],
        model_results = config["model_results_path"],
        data_path = config["sw-irr-output_path"]

    shell:
        """
        python fdl18_cdfg_residual_unified_test_to_tirr.py  \
        --src {configfiles} \
        --models {params.model_results} \
        --data_root {params.data_path} \
        --target {params.model_results} \
        # --eve_root {par} \
        --phase {params.phase}
        """



# ########################################################################################################
# #USE MODEL FOR INFERENCE
# ########################################################################################################

##
rule make_inference:
    input:
    output:
    shell:
        "python make_csv_inference.py"

##
rule unified_inference:
    input:
        means="eve_residual_mean_14ptot.npy",
        stds="eve_residual_std_14ptot.npy",
        trained_loss = "trained_loss.npy",
        val_loss = "val_loss.npy"
    output:
        eve_prediction= "eve_prediction.csv"
    shell:
        "python cdf_residual_unified_inference_totirr.py {output.eve_prediction}"