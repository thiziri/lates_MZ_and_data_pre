#!/usr/bin/env bash
cd ../../

currpath=`pwd`
# train the model
echo "mvlstm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/mvlstm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/mvlstm_quoraqp.log

echo "amvlstm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/amvlstm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/amvlstm_quoraqp.log

echo "matchpyramid .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/matchpyramid_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/matchpyramid.log

echo "knrm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/knrm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/knrm.log

echo "duet .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/duet_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/duet.log

echo "dssm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/dssm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/dssm.log

echo "drmm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/drmm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/drmm.log

echo "drmm_tks .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/drmm_tks_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/drmm_tks.log

echo "conv_knrm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/conv_knrm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/conv_knrm.log

echo "cdssm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/cdssm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/cdssm.log

echo "cdssm_word .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/cdssm_word_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/cdssm_word.log

echo "bimpm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/bimpm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/bimpm.log

echo "arci .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/arci_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/arci.log

echo "arcii .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/arcii_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/arcii.log

echo "anmm .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/anmm_quoraqp.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/anmm.log

echo "conv_weakColl .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/conv_weak_collaboration.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/conv_weak_collaboration.log

echo "siames_weak_coll .."
python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/weak_collaboration.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/QuoraQP/train_logs/weak_collaboration.log





