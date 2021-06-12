#!/bin/bash                                                                                                                                                                                                    
python compute_wor.py \
	--dataset=topical-usr \
	--device=cuda \
	--am_model_path=embedding_models/full_am \
	--fm_model_path=language_models/full_fm \
	--criterion Understandable Natural "Maintains Context" Engaging "Uses Knowledge" Overall

