# brain-llm-convergence
An analysis of the architectural and functional similarity between a language model and the brain. 



## Pipeline
The piepeline in this project is sketched out in the following Figure:

![brains_pipeline](https://github.com/user-attachments/assets/fe180f18-6fc9-4733-8eb1-c519f3c0893b)

The left-hand tile illustrates the groups of 4 words, curated by Wehbe et al. (2014), which are used as stimuli. The 2 bottom tiles outline the experimental process of fMRI data acquisition undertaken by Wehbe et al. (2014). The 2 top tiles indicate that I use Sentence Transformers (SBERT) (Reimers and Gurevych, 2019) to export embeddings and attention heads from all 6 layers of SBERT for every group of 4 words. The right-hand tile represents the linguistic features manually annotated for each group of 4 words by Wehbe et al. (2014). The arrow from the tile representing embeddings and attention heads to the tile representing linguistic features indicates that I am using ridge regression (β̂R) to analyse the representations of SBERT which track specific linguistic features. The arrow from the tile with the brain image to the tile representing linguistic features is used to show that I am using the FReM (Hoyos-Idrobo et al., 2018) algorithm (indicated with the magnifying glass icon) to analyse which parts of the brain are activated for specific linguistic features.

## Code
- **Wehbe_loader.py**: Loads and preprocesses data from Wehbe et al. (2014).
- **align_data.py**: Aligns brain activity and language model outputs for comparative analysis.
- **get_SBERT_embeddings.py**: Generates sentence-level embeddings using SBERT.
- **probing.py**: Trains, validates, and tests a ridge regression model on the embeddings to assess model interpretability. This script is adapted for computing on a GPU cluster.
- **decoder_frem_forward.py** & **decoder_frem_backward.py**: Implements Feature-Regression Models (FReM) to decode linguistic features from brain data. The code is divided into two to speed up the computations by using parallel computing. **decoder_frem_forward.py** is used for the first six brain regions, **decoder_frem_backward.py** is used for the remaining six brain regions. The division into groups is arbitrary, but it is necessary for the two groups to be disjoint sets if each group has half of the brain regions.
- **calculate_alignment.py**: Computes and visualizes alignment metrics (cosine similarity) between neural activity and model-generated features.

The probing and brain decoding code was executed using GPU clusters (2x RTX 3090 and 2x RTX 4090). Slurm scripts are not shared here. 

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References
- Andrés Hoyos-Idrobo, Gaël Varoquaux, Yannick Schwartz, and Bertrand Thirion. Frem - scalable and stable decoding with fast regularized ensemble of models. NeuroImage, 180:160–172, 2018. ISSN 1053-8119. doi: 10.1016/j.neuroimage.2017.10.005.
- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Lan-
guage Processing. Association for Computational Linguistics, 11 2019. URL https://arxiv.org/abs/1908.10084.
- Wehbe L, Murphy B, Talukdar P, Fyshe A, Ramdas A, et al. (2014) Simultaneously Uncovering the Patterns of Brain Regions Involved in Different Story Reading Subprocesses. PLoS ONE 9(11): e112575. doi:10.1371/journal.pone. 0112575
