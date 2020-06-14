settings = {
    'main_dirName': None,
    'time_scale' : None, # 24 for ICEWS dataset, 1 for GDELT dataset.
    'CI': None, # confidencial interval, .5 for ICEWS and 1 for GDELT.
    'time_horizon': 50,  # horizon by time prediction.
    'embd_rank': 200,  # hidden dimension of entity/rel embeddings
    'max_hist_len': 10, #maximum history sequence length for get_history
    'cut_pos': 10, #cuttoff position by prediction
}

