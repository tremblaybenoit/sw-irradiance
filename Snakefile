#########################################################################################################
##SETUP AND GENERATE TRAINING DATA
#########################################################################################################

configfile:"config_residuals/anet_3_bn_15.json"

# wavelengths = ["171","193","211","304"]

##finds corresponding eve data given path to aia files
rule generate_matches_time:
    input:
        eve_path = "/home/miraflorista/sw-irradiance/data/EVE/EVE.json",
        aia_path = "/mnt/miniset/aia",
        wavelength = "171"
    output:
        "/home/miraflorista/sw-irradiance/data/"
    shell:
        "python canonical_data/generate_matches_time.py -eve_path {input.eve_path} -aia_path {input.aia_path} -out_path {output[0]} -wavelengths input.wavelength "

## generates train, test, values datasets 
rule make_train_val_test_sets:
    input:
        matches = "/home/miraflorista/sw-irradiance/data/matches_aia_eve.csv"
    output:
        new_target="target.csv",
        new_model="model.csv",
    script:
        "canonical_data/make_train_val_test_sets.py"


rule make_normalize:
    input:
        matches = "/home/miraflorista/sw-irradiance/data/matches_aia_eve.csv"
    output:
        mean= "aia_sqrt_mean.npy",
        std= "aia_sqrt_std.npy"
    script:
        "canonical_data/make_normalize.py"

#########################################################################################################
##TEST AND TRAIN MODEL
#########################################################################################################

## fits means and stds to linear model
rule fit_linear_model:
    input:
        mean= "aia_sqrt_mean.npy",
        std= "aia_sqrt_std.npy",
        new_target="target.csv",
        new_model="model.csv"
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