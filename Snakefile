#########################################################################################################
##SETUP AND GENERATE TRAINING DATA
#########################################################################################################
configfile:"config_residuals/anet_3_bn_15.json"


## finds corresponding eve data given path to aia files
## make sure the output path matches your user home folder

rule create_eve_netcdf:
    input:
        eve_raw_path= "/home/miraflorista/sw-irradiance/data/EVE/raw/",
    output:
        eve_netcdf_path= "../sw-irr-output/EVE_irradiance.nc"
    shell:
        "python canonical_data/fdl22_create_eve_netcdf.py -eve_raw_path {input.eve_raw_path} -netcdf_outpath {output.eve_netcdf_path}"


## generates matches between EVE data and the AIA files
rule generate_matches_time:
    input:
        eve_netcdf_path = "../sw-irr-output/EVE_irradiance.nc",
        aia_path = "/mnt/aia-jsoc"
    params:
        match_outpath = "../sw-irr-output"
    output:
        matches_output = "../sw-irr-output/matches_eve_aia_171_193_211_304.csv"
    shell:
        "python canonical_data/fdl22_generate_matches_time.py -eve_path {input.eve_netcdf_path} -aia_path {input.aia_path} -out_path {params.match_outpath}"

## generates train, test, values datasets 
rule make_train_val_test_sets:
    input:
        matches="/home/andres_munoz_j/sw-irr-output/matches_eve_aia_171_193_211_304.csv"
    output:
        train = "/home/andres_munoz_j/sw-irr-output/train.csv",
        val = "/home/andres_munoz_j/sw-irr-output/val.csv",
        test = "/home/andres_munoz_j/sw-irr-output/test.csv"
    shell:
        "python canonical_data/fdl18_make_splits.py --src {input.matches} --splits rve"


# Creates the normalization values based on the train set
rule make_normalize:
    input:
        matches = "/home/andres_munoz_j/sw-irr-output/matches_eve_aia_171_193_211_304.csv",
    params:
        basepath = "/home/andres_munoz_j/sw-irr-output",
        divide = 4
    output:
        eve_mean= "/home/andres_munoz_j/sw-irr-output//eve_mean.npy",
        eve_std= "/home/andres_munoz_j/sw-irr-output//eve_std.npy",
        eve_sqrt_mean= "/home/andres_munoz_j/sw-irr-output//eve_sqrt_mean.npy",
        eve_sqrt_std= "/home/andres_munoz_j/sw-irr-output//eve_sqrt_std.npy",
        aia_mean= "/home/andres_munoz_j/sw-irr-output//aia_mean.npy",
        aia_std= "/home/andres_munoz_j/sw-irr-output//aia_std.npy",
        aia_sqrt_mean= "/home/andres_munoz_j/sw-irr-output//aia_sqrt_mean.npy",
        aia_sqrt_std= "/home/andres_munoz_j/sw-irr-output//aia_sqrt_std.npy"
    shell:
        "python canonical_data/fdl22_make_normalize.py --base /home/andres_munoz_j/sw-irr-output/ --divide {params.divide}"

#########################################################################################################
##TEST AND TRAIN MODEL
#########################################################################################################

## fits means and stds to linear model
rule fit_linear_model:
    params:
        basepath = "/home//sw-irr-output",
    input:
        eve_mean= "/home/andres_munoz_j/sw-irr-output//eve_mean.npy",
        eve_std= "/home/andres_munoz_j/sw-irr-output//eve_std.npy",
        eve_sqrt_mean= "/home/andres_munoz_j/sw-irr-output//eve_sqrt_mean.npy",
        eve_sqrt_std= "/home/andres_munoz_j/sw-irr-output//eve_sqrt_std.npy",
        aia_mean= "/home/andres_munoz_j/sw-irr-output//aia_mean.npy",
        aia_std= "/home/andres_munoz_j/sw-irr-output//aia_std.npy",
        aia_sqrt_mean= "/home/andres_munoz_j/sw-irr-output//aia_sqrt_mean.npy",
        aia_sqrt_std= "/home/andres_munoz_j/sw-irr-output//aia_sqrt_std.npy"
    output:
        means="eve_residual_mean_14ptot.npy",
        stds="eve_residual_std_14ptot.npy"
    shell:
        "python canonical_code/setup_residual_totirr.py"

## train CNN
rule train_CNN:
    input:
        means="eve_residual_mean_14ptot.npy",
        stds="eve_residual_std_14ptot.npy"
    output:
        trained_loss = "trained_loss.npy",
        val_loss = "val_loss.npy"
    shell:
        "python cdfg_residual_unified_train_to_tirr.py {input.config_file} {input.means}"


## test data
rule test_CNN:
    input:
        trained_loss = "trained_loss.npy",
        val_loss = "val_loss.npy"
    output:
        errors = "errors.npy"
    shell:
        "python cdfg_residual_unified_test_to_tirr.py"



# ########################################################################################################
# #USE MODEL FOR INFERENCE
# ########################################################################################################

##
rule make_inference:
    input:
    output:
    shell:
        "python make_csv_inference.csv"

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