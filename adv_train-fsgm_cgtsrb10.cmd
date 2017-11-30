
for /l %%x in (1, 1, 10) do (
    python run.py adv_fgsm cgtsrb10
    echo %%x >> adv_training_cgtsrb10-fgsm.txt
    python dump_adv_success_rate.py >> adv_training_cgtsrb10-fgsm.txt
    python run.py adv_train cgtsrb10
    del tmp\adv_cgtsrb10-32x32\* /q
)