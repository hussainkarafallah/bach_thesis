
datasets_params = {
    'ml-100k' : {
        'min_user_inter_num' : 5,
        'min_item_inter_num' : 5,
        'eval_setting': 'TO_RS,full',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric' : 'Recall@20',
        "split_ratio" : [0.6,0.2,0.2],
        'eval_batch_size' : 1000000
    } ,
    'gowalla' : {
        'load_col' : {'inter': ['user_id', 'item_id','timestamp']},
        'min_user_inter_num' : 10,
        'min_item_inter_num' : 10,
        'eval_setting': 'TO_RS,full',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric' : 'Recall@20',
        'split_ratio' : [0.7 , 0.1 , 0.2],
        'eval_batch_size': 1000000
    } ,
    'ml-1m' : {
        'min_user_inter_num' : 5,
        'min_item_inter_num' : 5,
        'eval_setting': 'TO_RS,full',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp' , 'rating']},
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric' : 'Recall@20',
        "split_ratio" : [0.7,0.1,0.2],
        'eval_batch_size' : 1000000,
        'lowest_val' : { 'rating' : 3 }
    } ,
    'lastfm' : {
        'load_col': {'inter': ['user_id', 'artist_id']},
        'min_user_inter_num' : 5,
        'min_item_inter_num' : 5,
        'eval_setting': 'RO_RS,full',
        'metrics' : ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 10,
        'valid_metric' : 'Recall@10',
        "split_ratio" : [0.6,0.2,0.2],
        'eval_batch_size' : 1000000,
        'ITEM_ID_FIELD': 'artist_id',
    }
}
