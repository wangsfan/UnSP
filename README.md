# UnSP

The source code of **UnSP: Improving Event-to-Image Reconstruction with Uncertainty Guided Self-Paced Learning**


## Training Dataset
You will need to generate the training dataset yourself, using ESIM. To find out how, please see the training data generator repo(https://github.com/TimoStoff/esim_config_generator).
## Testing Dataset
One can download the testing dataset(IJRR17) from the link below:
https://download.ifi.uzh.ch/rpg/web/data/E2VID/datasets/ECD_IJRR17/. 
## Train
python self_paced_test.py
## Test
python my_benchmark.py

# Citation

If you compare to this model or use this code for any means, please cite us! 

@article{yang2025unsp,
	title={UnSP: Improving Event-to-Image Reconstruction with Uncertainty Guided Self-Paced Learning},
	author={Yang, Jianye and Zhang, Xiaolin and Wang, Shaofan and Sun, Yanfeng and Yin, Baocai},
	journal={Displays},
	year = {2025},
	volume={87},
	pages = {102985:1-102985:9}
}
