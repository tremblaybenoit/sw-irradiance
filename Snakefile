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
        eve_netcdf_path= config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/EVE_irradiance.nc"
    shell:
        "python fdleuvai/data/preprocess/create_eve_netcdf.py \
        -eve_raw_path {input.eve_raw_path} \
        -netcdf_outpath {output.eve_netcdf_path}"


## generates matches between EVE data and the AIA files
rule generate_matches_time:
    input:
        eve_netcdf_path = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/EVE_irradiance.nc",
        aia_path = config["aia_path"]
    params:
        match_outpath = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]),
        debug = config["DEBUG"]
    output:
        matches_output = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/matches_eve_aia_171_193_211_304.csv"
    shell:
        """
        python fdleuvai/data/preprocess/generate_matches_time.py \
        -eve_path {input.eve_netcdf_path} \
        -aia_path {input.aia_path} \
        -out_path {params.match_outpath}\
        -debug {params.debug}
        """

## generates donwnscaled stacks of the AIA chanels
rule generate_euv_image_stacks:
    input:
        matches = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/matches_eve_aia_171_193_211_304.csv",
        aia_path = config["aia_path"]
    params:
        stack_outpath = config["aia_stack_path"]+"-"+str(config["RESOLUTION"]),
        resolution = config["RESOLUTION"],
        remove_off_limb = config["REMOVE_OFFLIMB"],
        debug = config["DEBUG"]
    output:
        matches_output = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/matches_eve_aia_171_193_211_304_stacks.csv"
    shell:
        """
        python fdleuvai/data/preprocess/generate_euv_image_stacks.py \
        -aia_path {input.aia_path} \
        -matches {input.matches}\
        -stack_outpath {params.stack_outpath}\
        -resolution {params.resolution}\
        -remove_off_limb {params.remove_off_limb}\
        -debug {params.debug}
        """

## generates train, test, values datasets 
rule make_train_val_test_sets:
    input:
        matches = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/matches_eve_aia_171_193_211_304_stacks.csv"
    output:
        expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/{split}.csv",split = config["SPLIT"])
    shell:
        "python fdleuvai/data/preprocess/make_train_val_test_sets.py -src {input.matches} -splits rve"
        

#########################################################################################################
##TEST AND TRAIN MODEL
#########################################################################################################

## fits means and stds to linear model
rule fit_linear_model:
    params:
        basepath = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]),
        resolution = config["RESOLUTION"],
        remove_off_limb = config["REMOVE_OFFLIMB"],
        debug = config["DEBUG"]
    input:
        matches = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/matches_eve_aia_171_193_211_304_stacks.csv",
        splits = expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/{split}.csv",split = config["SPLIT"]),
        eve_netcdf_path=config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/EVE_irradiance.nc"
    output:
        linear_preds =expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/EVE_linear_pred_{split}.nc",split = config["SPLIT"]),
        # linear_stats =config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/mean_std_feats.npz",
        # eve_resid =config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/eve_residual_{}_14ptot.npy"

    shell:
        """
        python fdleuvai/models/fit_linear_model.py \
        -base {params.basepath} \
        -resolution {params.resolution}\
        -remove_off_limb {params.remove_off_limb}\
        -debug {params.debug}
        """

## Creates the normalization values based on the train set
rule calculate_training_normalization:
    input:
        matches = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/matches_eve_aia_171_193_211_304_stacks.csv",
        splits = expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/{split}.csv",split = config["SPLIT"]),
        eve_netcdf_path = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/EVE_irradiance.nc"
    params:
        basepath = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]),
        resolution = config["RESOLUTION"],
        remove_off_limb = config["REMOVE_OFFLIMB"],
        debug = config["DEBUG"]
    output:
        eve_norm_stats=expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]) + "/eve_{norm_stat}.npy", norm_stat=config["EVE-NORM-STATISTIC"]),
        aia_norm_stats=expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]) + "/aia_{norm_stat}.npy", norm_stat=config["AIA-NORM-STATISTIC"])
    shell:
        """
        python fdleuvai/data/preprocess/calculate_training_normalization.py \
        -base {params.basepath} \
        -resolution {params.resolution}\
        -remove_off_limb {params.remove_off_limb}\
        -debug {params.debug}
        """

## train CNN
rule train_CNN:
    input:
        eve_netcdf_path = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/EVE_irradiance.nc",
        matches = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/matches_eve_aia_171_193_211_304_stacks.csv",
        splits = expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/{split}.csv",split = config["SPLIT"]),
        linear_preds =expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])+"/EVE_linear_pred_{split}.nc",split = config["SPLIT"]),
        eve_norm_stats=expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]) + "/eve_{norm_stat}.npy", norm_stat=config["EVE-NORM-STATISTIC"]),
        aia_norm_stats=expand(config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]) + "/aia_{norm_stat}.npy", norm_stat=config["AIA-NORM-STATISTIC"])
    output:
        trained_loss = config["model_results_path"] + "/" + config["nn_config_file"] + "/train_loss.npy",
        val_loss = config["model_results_path"] + "/" + config["nn_config_file"] + "/val_loss.npy",
        model = expand(config["model_results_path"] + "/" + config["nn_config_file"] + "/" + config["nn_config_file"] + "_{train_outputs}", train_outputs=config["TRAIN_OUTPUTS"])        
    params:
        data_path = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"]),
        model_results = config["model_results_path"],
        config_path = config["nn_config_path"],
        model_results_path = config["model_results_path"],
        remove_off_limb = config["REMOVE_OFFLIMB"],
        debug = config["DEBUG"]
    shell:
        """
        python fdleuvai/models/fit_NN_models.py\
        -src {params.config_path}\
        -data_root {params.data_path}\
        -target {params.model_results_path}\
        -remove_off_limb {params.remove_off_limb}\
        -debug {params.debug}
         """

## test data
rule test_CNN:
    input:
        trained_loss = "trained_loss.npy",
        val_loss = "val_loss.npy"
    output:
        errors = "errors.npy"
    params:
        phase = config["PHASE"],
        model_results = config["model_results_path"],
        data_path = config["sw-irr-output_path"]+"-"+str(config["RESOLUTION"])

    shell:
        """
        python fdl18_cdfg_residual_unified_test_to_tirr.py  \
        -src {configfiles} \
        -models {params.model_results} \
        -data_root {params.data_path} \
        -target {params.model_results} \
        # -eve_root {par} \
        -phase {params.phase}
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