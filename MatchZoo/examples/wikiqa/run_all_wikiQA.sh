cd ../../

currpath=`pwd`
# train the model
echo "mvlstm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/mvlstm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/mvlstm_wikiqa.log

echo "matchpyramid .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/matchpyramid_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/matchpyramid.log

echo "knrm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/knrm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/knrm.log

echo "duet .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/duet_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/duet.log

echo "dssm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/dssm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/dssm.log

echo "drmm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/drmm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/drmm.log

echo "drmm_tks .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/drmm_tks_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/drmm_tks.log

echo "conv_knrm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/conv_knrm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/conv_knrm.log

echo "cdssm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/cdssm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/cdssm.log

echo "cdssm_word .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/cdssm_word_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/cdssm_word.log

echo "bimpm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/bimpm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/bimpm.log

echo "arci .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/arci_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/arci.log

echo "arcii .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/arcii_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/arcii.log

echo "anmm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/anmm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/anmm.log

echo "conv_weakColl .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/conv_weak_collaboration.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/conv_weak_collaboration.log

echo "siames_weak_coll .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/weak_collaboration.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/weak_collaboration.log





