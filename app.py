import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nmt import nmt, attention_model, model_helper
import tensorflow as tf
from nmt.utils import misc_utils
from nmt.utils import nmt_utils
import argparse
from nmt import utils
from flask import Flask, jsonify, render_template, request
import json

app = Flask(__name__)
model_dir = "friends/nmt_model"
FLAGS = None
nmt_parser = argparse.ArgumentParser()
nmt.add_arguments(nmt_parser)
FLAGS, unparsed = nmt_parser.parse_known_args()

def tokenize_text(text):
	tokens = word_tokenize(text)
	return ' '.join(tokens).strip()

def generate_reply(input_text, flags):
	# Format data
	tokenized_text = tokenize_text(input_text)
	infer_data = [tokenized_text]	
	# Load hparams.
	jobid = flags.jobid
	default_hparams = nmt.create_hparams(flags)
	hparams = nmt.create_or_load_hparams(
		model_dir, default_hparams, flags.hparams_path, save_hparams=(jobid == 0))
	# Load checkpoint
	ckpt = tf.train.latest_checkpoint(model_dir)
	# Inference
	model_creator = attention_model.AttentionModel
	# Create model
	scope = None
	infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

	with tf.Session(graph=infer_model.graph, config=misc_utils.get_config_proto()) as sess:
		model = model_helper.load_model(infer_model.model, ckpt, sess, "infer")
		sess.run(infer_model.iterator.initializer,
			feed_dict={
			infer_model.src_placeholder: infer_data,
			infer_model.batch_size_placeholder: hparams.infer_batch_size})
		# Decode
		nmt_outputs, infer_summary = model.decode(sess)
		# get text translation(reply as a chatbot)
		assert nmt_outputs.shape[0] == 1
		translation = nmt_utils.get_translation(
			nmt_outputs,
			sent_id=0,
			tgt_eos=hparams.eos,
			subword_option=hparams.subword_option)

	return translation.decode("utf-8")

def pred(input_text):
	if not input_text: return
	answer = generate_reply(input_text, FLAGS)
	return answer

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
	if request.json['message'] is None:
		return jsonify('json error')
	response = pred(str(request.json['message']))
	return jsonify(response)

@app.route('/')
def main():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)