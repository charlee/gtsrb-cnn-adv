REM python run.py adv_fast_jsma mnistbg
REM python run.py adv_fast_jsma cgtsrb10
REM python run.py adv_fast_jsma fmnist
REM python run.py adv_fast_jsma cifar10
REM python run.py adv_fgsm mnist
REM python run.py adv_fast_jsma mnist

REM python run.py train cgtsrb10
REM python run.py train fmnist
REM python run.py train mnistbg
REM python run.py train cifar10
REM python run.py train mnist
REM python run.py adv_fast_jsma mnist


python run.py adv_fgsm_train fmnist
python run.py adv_fgsm_train cgtsrb10
python run.py adv_fgsm_train mnistbg
python run.py adv_fgsm_train cifar10
python run.py adv_fgsm_train mnist

python run.py adv_fgsm cgtsrb10
python run.py adv_fgsm fmnist
python run.py adv_fgsm mnistbg
python run.py adv_fgsm cifar10
python run.py adv_fgsm mnist