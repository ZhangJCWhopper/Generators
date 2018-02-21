import model_test_kits
import os

tool = model_test_kits.PreparePLDA()
#First, make train 10sec generated
tool.short_ivs_file = "exp/ivectors_sre08_train_10sec_male/spk_ivector.ark"
tool.eval_space = "/files/eval"
tool.target_ark_file = tool.target_ark_file.replace(".ark", "_train.ark")

model_dir = "wcgan_results28"

tool.generate_samples_light(model_dir)

tool.short_ivs_file = "exp/ivectors_sre08_test_10sec_male/ivector.ark"
tool.target_ark_file = tool.target_ark_file.replace("train.ark", "test.ark")

tool.generate_samples_light(model_dir)


"""
trials=10sec-10sec-male.trials
cat exp/ivectors_sre08_train_10sec_male/spk_ivector.scp exp/ivectors_sre08_test_10sec_male/ivector.scp > male.scp

ivector-plda-scoring --simple-length-normalization=true --num-utts=ark:exp/ivectors_sre08_train_10sec_male/num_utts.ark / 
"ivector-adapt-plda exp/plda_male scp:male.scp -|" ark:exp/ivectors_sre08_train_10sec_male/spk_ivector.ark ark:exp/ivectors_sre08_test_10sec_male/ivector.ark / 
"cat '$trials' | awk '{print \$1, \$2}' |" 10sec_foo


cat eval/*_LN.scp > eval/10sec_male.scp

ivector-plda-scoring --simple-length-normalization=true --num-utts=ark:exp/ivectors_sre08_train_10sec_male/num_utts.ark / 
"ivector-adapt-plda exp/plda_male scp:eval/10sec_male.scp -|" ark:eval/generated_ivectors_LN_train.ark ark:eval/generated_ivectors_LN_test.ark / 
"cat '$trials' | awk '{print \$1, \$2}' |" 10sec_gen_foo
"""