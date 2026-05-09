cd ..

export WANDB_API_KEY="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"

python genrec/trainers/augr_trainer.py config/hstu/amazon_augr.gin --split sports &
python genrec/trainers/augr_trainer.py config/hstu/amazon_augr.gin --split beauty &
python genrec/trainers/augr_trainer.py config/hstu/amazon_augr.gin --split toys &

wait