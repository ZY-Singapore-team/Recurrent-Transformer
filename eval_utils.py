from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

from bert_serving.client import BertClient

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: coco-caption not available')

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    print(dataset)
    if 'val' in dataset:
        annFile = 'coco-caption/annotations/para_captions_val.json'
    if 'test' in dataset:
        annFile = 'coco-caption/annotations/para_captions_test.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    return COCO(annFile)


def language_eval(dataset, preds, eval_kwargs, split):
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)
    
    # create output dictionary
    out = {}
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')
    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    # for p in preds_filt:
    #     image_id, caption = p['image_id'], p['caption']
    #     imgToEval[image_id]['caption'] = caption

    # print prediction - gts file
    imgs = json.load(open('data/captions/para_karpathy_format.json', 'r'))
    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
        img = [x for x in imgs['images'] if x['id'] == image_id][0]
        imgToEval[image_id]['gts_caption'] = img['sentences'][0]['raw']

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    sent_num = eval_kwargs.get('seq_per_img', 6)
    rnn_size = eval_kwargs.get('rnn_size', 512)
    topic_vec_size = eval_kwargs.get('topic_vec_size', 768)
    decode_sent_num = sent_num
    # decode_sent_num = 4

    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['topics']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks, topic_vecs= tmp

            #### Liangming: Add a for loop here
            labels = labels.view(att_feats.shape[0], sent_num, -1)
            masks = masks.view(att_feats.shape[0], sent_num, -1)
            # topic_vecs = topic_vecs.view(att_feats.shape[0], sent_num, -1)
            topic_vec = torch.zeros((att_feats.shape[0], labels.shape[2] - 1 , rnn_size)).float().cuda()

            for sent_n in range(decode_sent_num):
                # prepare sentence data
                sent_label = labels[:, sent_n, :]
                sent_mask = masks[:, sent_n, :]

                # We should skip the batch in which the sentences for all examples in the batch are 0s. 
                if torch.sum(sent_label).item() == 0:
                    continue

                with torch.no_grad():
                    # import ipdb; ipdb.set_trace()
                    predicts, decoder_out = model(fc_feats, att_feats, sent_label, topic_vec, att_masks)
                    loss = crit(predicts, sent_label[:,1:], sent_mask[:,1:]).item()
                loss_sum = loss_sum + loss
                loss_evals = loss_evals + 1

                # update topic vec
                # topic_vec = decoder_out

                # Optional: Shrink the size of it based on the mask?
                max_sent_len = -1
                for row in range(sent_mask.shape[0]):
                    sent_len = int(sum(sent_mask[row, :]).data.item())
                    if sent_len > max_sent_len:
                        max_sent_len = sent_len
                topic_vec = decoder_out[:, 0:max_sent_len, :]

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'], 
            data['att_feats'],
            data['att_masks']]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp

        ### Inference Codes
        ### Liangming: when t = 0, pass topic_vec inside (all zeros)
        ### when t != 0, calculate the new topic_vec based on decoded sequence, and then pass it to the next step

        # Initilize topic vector
        decode_seq_len = 31
        topic_vec = torch.zeros((att_feats.shape[0], decode_seq_len, rnn_size)).float().cuda()
        # the final results
        paragraph = []
        for ex in range(att_feats.shape[0]):
            paragraph.append('')

        #### Liangming: Add a for loop here
        for sent_n in range(decode_sent_num):
            # For each sentence
            with torch.no_grad():
                tmp_eval_kwargs = eval_kwargs.copy()
                tmp_eval_kwargs.update({'sample_n': 1})
                # Liangming: Add topic vec
                seq, seq_logprobs, decoder_out = model(fc_feats, att_feats, topic_vec, att_masks, opt=tmp_eval_kwargs, mode='sample')
                # update decoder output
                decoder_states = decoder_out[0].unsqueeze(1)
                for i in range(1, len(decoder_out)):
                    decoder_states = torch.cat((decoder_states, decoder_out[i].unsqueeze(1)), 1)
                # update topic vecs
                topic_vec = decoder_states

                seq = seq.data
                entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq>0).float().sum(1)+1)
                perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq>0).float().sum(1)+1)
            
            # Print beam search
            if beam_size > 1 and verbose_beam:
                for i in range(fc_feats.shape[0]):
                    print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                    print('--' * 10)
            sents = utils.decode_sequence(loader.get_vocab(), seq)
            # sents is a list of batch_size, each element is a decoded sentence (string)
            # update sents to graph
            for ind, sent in enumerate(sents):
                paragraph[ind] += (' ' + sent)

            # import ipdb; ipdb.set_trace()
            ### IMP: perpare and update topic vector based on the decoded sequence.
            # bc = BertClient()
            # corpus = [sent.split() for sent in paragraph]
            # topic_vecs_numpy = bc.encode(corpus, is_tokenized=True)
            # topic_vec = topic_vec.new_tensor(topic_vecs_numpy)
            # import ipdb; ipdb.set_trace()
            ######

        for k, sent in enumerate(paragraph):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            # if verbose:
            #     print('image %s: %s' %(entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_split_n(model, n_predictions, loader, [fc_feats, att_feats, att_masks, data], eval_kwargs)
        
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs, split)
    
    Top_K = 5
    print('Sample Predictions:')
    for i in range(Top_K):
        entry = predictions[i]
        print('\timage %s: %s' %(entry['image_id'], entry['caption']))
        
    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


# Only run when sample_n > 0
def eval_split_n(model, n_predictions, loader, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, att_masks, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(loader.get_vocab(), torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update({'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1}) # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(loader.get_vocab(), _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / ((_seq>0).float().sum(1)+1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent, 'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(loader.get_vocab(), torch.stack([model.done_beams[k][_]['seq'] for _ in range(0, sample_n*beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update({'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size':1}) # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(loader.get_vocab(), _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-loader.batch_size * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' %(entry['image_id'], entry['caption']))