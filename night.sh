nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation +cluster=slurm >> night1.log 2>&1 &
nohup python main.py --multirun +experiment=multimodal_PBMC_bknn_augmentation_ablation +cluster=slurm >> night2.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation_qr +cluster=slurm >> night3.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_PBMC_bknn_augmentation_ablation_qr +cluster=slurm >> night4.log 2>&1 &

nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation +cluster=slurm >> night1.log 2>&1 & &
nohup python main.py --multirun +experiment=multimodal_Neurips_bknn_augmentation_ablation +cluster=slurm >> night2.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_qr +cluster=slurm &
# nohup python main.py --multirun +experiment=multimodal_Neurips_bknn_augmentation_ablation_qr +cluster=slurm &
