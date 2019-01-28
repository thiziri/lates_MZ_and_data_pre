# ref: https://www.datacamp.com/community/tutorials/python-xml-elementtree

import xml.etree.ElementTree as ET
import sys
import json
import logging
import os
from os.path import join
from xml.etree import ElementTree
from xml.dom import minidom
from tools4text import extract_trec_million_queries, clean, read_values
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import stopwords

def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ").replace("<?xml version=\"1.0\" ?>\n", "")


logging.basicConfig(filename="logs/parse_xml_parameters4indri.log", level=logging.DEBUG)

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = json.load(open(config_file))
    logging.info('Config: '+json.dumps(config, indent=2))
    out = config["output"]

    # process queries:
    queries = extract_trec_million_queries(config["queries"])
    queries_text = {}
    stoplist = set(stopwords.words("english")) if config["stopwords"] else {}
    qrels_MQ = []  # get only judged queries
    for file in os.listdir(config["qrels_MQ"]):
        qrels_MQ += list(read_values(os.path.join(config["qrels_MQ"], file), 0))
    qrels_MQ = set(qrels_MQ)
    q_times = defaultdict(int)
    print("Pre-process queries %d queries..." % len(qrels_MQ))
    logging.info("Pre-process queries %d queries..." % len(qrels_MQ))
    for q in tqdm(qrels_MQ):
        q_text = clean(queries[q], config["stemmer"], stoplist)
        q_times[q_text] += 1  # queries with duplicate content
        queries_text[q] = q_text if q_times[q_text] == 1 else ' '.join([q_text, str(q_times[q_text])])
        if q_times[q_text] > 1:
            logging.info("Duplicated: " + q + '\t' + ' '.join([q_text, str(q_times[q_text])]))

    # retrieve passages
    logging.info("Retrieving ...")
    print("Wait while retrieving passages for different queries ...")
    temp_parameters = join(config["output"], "temp_Q.xml")
    for q_id in qrels_MQ:
        q_txt = queries_text[q_id]
        logging.info(q_id + '\t' + q_txt)
        c_root = ET.Element("parameters")
        c_query = ET.SubElement(c_root, "query")
        type_ = ET.SubElement(c_query, "type")
        type_.text = "indri"
        num = ET.SubElement(c_query, "number")
        num.text = q_id
        text = ET.SubElement(c_query, "text")
        text.text = "#combine[passage{pl}:{pw}]({q_txt})".format(pl=config["passage_length"],
                                                                 pw=config["sequence_length"], q_txt=q_txt)

        # save temporal xml file
        q_out = open(temp_parameters, 'w')
        q_out.write(prettify(c_root))
        q_out.close()

        # retrieve with INDRI:
        os.chdir(config["indri"])
        passages = os.popen("./runquery/IndriRunQuery {param} -count={c} -index={i}".format(param=temp_parameters,
                                                                                            c=config["count"],
                                                                                            i=config["index"])
                            ).read()
        out_passages = open(join(config["output"], q_id), 'w')
        out_passages.write(passages)
        out_passages.close()

    os.remove(temp_parameters)
    print("Finished.")
