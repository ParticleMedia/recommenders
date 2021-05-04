"""Global settings and imports"""
import sys
sys.path.append("../../")
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages
import json

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

"""Prepare parameters"""
epochs = 5
seed = 42
batch_size = 10

# Options: demo, small, large
MIND_type = 'demo'

"""Download and load data"""
data_path = os.path.join(os.path.dirname(__file__), "data")

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)

"""Create hyper-parameters"""
hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10,
                          run_eagerly=True)
hparams.dropout = 0.0
print(hparams)


"""Train the NRMS model"""
iterator = MINDIterator
hparams.dropout = 0.0
model = NRMSModel(hparams, iterator, seed=seed)
m = {}
model.run_eval(valid_news_file, valid_behaviors_file, -1, 60, m)
out_dir = sys.argv[1]
out_f = open(f"{out_dir}/tf_nrms_in_out_60_nomask.json", "w", encoding="UTF8")
m_str = json.dumps(m)
out_f.write(m_str)
out_f.close()

