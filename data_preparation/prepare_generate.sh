#!/usr/bin/env bash
# Prepare all data for MatchZoo
echo "Reading dataset index ..."
# python3 preparation/prepare_MZ_input_text.py config/prepare_MZ_input_text.config

echo "Preparation for ranking..."
cd /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/MatchZoo/examples/trec
python3 test_preparation_for_ranking.py /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/ fold_0

echo "Generate embeddings ..."
python gen_w2v.py /home/thiziri/Documents/DOCTORAT/COLLECTION/Embeddings/extended_AP88_skipgram_wordEmbedding_dim300_win10_minCount5.txt /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/word_dict.txt /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/AP88_skipgram_300wordEmbedding
python norm_embed.py /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/AP88_skipgram_300wordEmbedding /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/AP88_skipgram_300wordEmbedding_norm

echo 'Generate histograms ...'
python3 test_histogram_generator.py 30 300 /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/ /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/AP88_skipgram_300wordEmbedding_norm
wait 

echo 'Compute embeddings idf ...'
python3 /home/thiziri/Documents/DOCTORAT/PROGRAMS/Projects/2ndYear/data_preparation/for_matchZoo/utils/embed_idf.py --i=/home/thiziri/Documents/DOCTORAT/COLLECTION/Indri_index/AP88/ --d=/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/word_dict.txt --o=/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/data/AP88/from_qrels/same_ts_tr_vl/rank_qrels/fold_0/
wait

echo "Done."
