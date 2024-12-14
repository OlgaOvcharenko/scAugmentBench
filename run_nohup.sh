# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation +cluster=slurm >> PBMC_bc.log 2>&1 &
nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation_qr +cluster=slurm >> PBMC_qr.log 2>&1 &

# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation +cluster=slurm >> Neurips_bc.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_qr +cluster=slurm >> Neurips_qr.log 2>&1 &
