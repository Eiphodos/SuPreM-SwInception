# FAQ

* **Q: Due to the limited CPU memory, I can only cache 60% of the data. I am curious to know if reducing the cache number or the cache rate would significantly impact the final performance.**

    Short answer is yes, it will impact the performance, but you could increase the number of epochs to let the model train more.
    
    This is a very important question. I predict that more and more medical imaging researchers will meet the same issue shortly. Our AbdomenAtlas 1.0 [[Qu et al., NeurIPS 2023]](https://www.cs.jhu.edu/~alanlab/Pubs23/qu2023abdomenatlas.pdf) provides 5,000+ CT volumes, and our next version AbdomenAtlas 1.1 [[Li et al., ICLR 2024]](https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf) will provide 9,000+ volumes, and we also have a bigger version at Hopkins, called AbdomenAtlas Pro with 22,000 volumes.
    
    Dealing with increasing medical data/annotations will be the emerging research frontier for more exploration. We have already made several explorations on this (e.g., lifelong learning, better sampling strategies, better CPU-GPU communication, etc.). For this specific benchmark, you are allowed to modify any hyperparameters and use any engineering tricks to make your backbone competitive.

* **Q: What are the immediate implications for the average hospital patient?**

    Trained on our dataset, AI holds the premise to improve organ volume measurement accuracy and reduce manual contouring efforts. Precise organ volume measurement is fundamental for effective patient care, but manual organ contouring is extremely time-consuming and exhibits considerable variability among expert radiologists. Variations in organ sizes and shapes can indicate a range of medical conditions, from benign anomalies to life-threatening diseases. Our dataset provides detailed per-voxel annotations for eight abdominal organs. These can be used to develop AI that can automatically identifies and delineates the boundary of various anatomical structures, essential for numerous downstream applications such as disease diagnosis and treatment planning. Besides, as said in 5th answer, our dataset can be a great resource to improve AI performance of cancer-related tasks.

* **Q: How do the models trained on the collected large CT dataset generalize to novel modalities such as MRI?**

    This is a very good point. We think transfer learning across different imaging modalities, such as from CT to MRI, might be less effective compared to transfers within the same modality, primarily due to the significant differences in their imaging techniques. The discrepancies in image acquisition methods between CT and MRI result in distinct intensity values and ranges. Nonetheless, our pre-trained model could still be valuable for abdominal MRI applications. This is because the underlying anatomical structures remain consistent across both CT and MRI, allowing for the potential transfer of shared knowledge.

    Given the constraints of time, we are unable to include this specific experiment in the rebuttal period. However, we plan to incorporate a study focused on abdominal MRI tasks in the final version of our paper. With the release of our ImageNetCT-9K and SuPreM, we look forward to promoting a collaborative effort to thoroughly evaluate the capabilities of transfer learning. This includes exploring a wider range of medical modalities, such as T1, T1c, T2, Flair, and Ultrasound, and extending into general 3D vision tasks involving diverse data formats like point clouds, voxel occupancy grids, meshes, and implicit surface models (e.g., signed distance functions). 

* **Q: There exist many promising domain transfer methods, yet the paper appears to lack exploration in this area. The current approach appears to be straightforward fine-tuning on other datasets, such as TotalSegmentator and JHH.**

    The current approach is pretty robust (evidenced in Table 3) because our dataset covers a variety of domains (i.e., 68 hospitals with different scanners and protocols). Therefore, models pre-trained on this dataset are expected to be generalizable for novel domains, e.g., TotalSegmentator, FLARE’23, and JHH. These three datasets are completely unseen during the pre-training and represent a variety of diversity. Specifically, the TotalSegmentator dataset represents for the Central European population from Switzerland, the FLARE’23 dataset represents for East Asian population from China, and the JHH dataset represents for another population (anonymous for peer review). Our models achieve comparable or even superior performance to the IID counterparts (Table 3). Therefore, domain transfer becomes less important if the model is pre-trained on large and diverse datasets (elaborated in the next two points).

    1. The domain transfer problem could be solved by methodology innovation, as you suggested, and also by training AI models on enormous datasets. This point has been more clear recently demonstrated by large language models (GPT) and vision foundation models (SAM), which show incredible performance in “new domain”. However, this achievement may not be directly attributed to method-driven solutions for domain transfer, but simply because the AI might have been trained on similar sentences or images. This was also pointed out by Yann Lecun—“beware of testing on the training set”—in response to the incredible results achieved by GPT.
    
    2. In some sense, our paper explores dataset-driven solutions for domain transfer. The robust performance of our models when direct inference on multiple domains could also be attributed to our large-scale, fully-annotated medical dataset—as one of our major contributions. The release of this dataset can foster AI models that are more robust than the majority of existing models that are only trained on a few hundred CT volumes from limited domains. We completely agree that existing domain transfer methods could be supplemented with direct inference and fine-tuning to further improve AI performance.