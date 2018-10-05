cd /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest/MatchZoo

echo "Training begin ..."

# python3 matchzoo/main.py --phase train --model_file examples/wikiqa/config/sbdecatten_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/sbdecatten_attQ_wikiqa.log

#python3 matchzoo/main.py --phase train --model_file examples/wikiqa/config/matchpyramid_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/matchpyramid_attQ_wikiqa.log

python3 matchzoo/main.py --phase train --model_file examples/wikiqa/config/knrm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/knrm_attQ_wikiqa.log

python3 matchzoo/main.py --phase train --model_file examples/wikiqa/config/arci_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/arci_attQ_wikiqa.log

python3 matchzoo/main.py --phase train --model_file examples/wikiqa/config/arcii_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/arcii_attQ_wikiqa.log

python3 matchzoo/main.py --phase train --model_file examples/wikiqa/config/cdssm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/cdssm_attQ_wikiqa.log

python3 matchzoo/main.py --phase train --model_file examples/wikiqa/config/conv_knrm_wikiqa.config > /home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo_latest1/MatchZoo/examples/wikiqa/train_logs/conv_knrm_attQ_wikiqa.log

echo "Training finished."
