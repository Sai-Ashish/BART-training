#!/usr/bin/env python
# coding: utf-8
from rouge import Rouge
import torch
import time
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import datasets
import logging
from datasets import load_dataset
import utils
import glob
import sys
import os
from torch.autograd import Variable
import math
from transformers import get_scheduler
import argparse
import matplotlib.pyplot as plt

#####
# hyperparameters for training

parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--gradient_accumulation_steps', default=10, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')#change to 1e-2 if needed
parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer type')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--article_length', default=512, type=int, help='Article length')
parser.add_argument('--summary_length', default=60, type=int, help='Summary length')
parser.add_argument('--train_num', default=1000, type=int, help='Train number')
parser.add_argument('--lr_scheduler_type', type=str, default='polynomial', help='Learning rate scheduler type')
parser.add_argument('--num_warmup_steps', default=1000, type=int, help='num_warmup_steps')
parser.add_argument('--clip', default= 1.0, type=float)

#####
args = parser.parse_args()
#####

torch.cuda.set_device(0)

rouge = Rouge()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

dataset = load_dataset('xsum')


# ### Tokenize

from transformers import BartTokenizer, BartForConditionalGeneration
# Load the BART tokenizer.
logging.info('Loading BART tokenizer...')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
vocab_size = tokenizer.vocab_size

# Import the model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
model = model.cuda()
model.train()

# The loss function
def loss_fn(lm_logits, labels):
    loss_fct = CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
    loss = loss_fct(lm_logits.view(-1, vocab_size), labels.view(-1))
    return loss

def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']
    
    attention_mask = encoding['attention_mask']
    
    return input_ids, attention_mask

def get_dataset(dataset, num_points, DS_tokenizer, mode = 'validation'):
    
    print(dataset)
    
    # get the training data
    sentence = dataset[mode]['document'][:num_points]
    target = dataset[mode]['summary'][:num_points]
    
    # tokenize the article using the bart tokenizer
    article_input_ids, article_attention_mask = tokenize(sentence, DS_tokenizer, max_length = args.article_length)
    print("Input shape: ")
    print(article_input_ids.shape, article_attention_mask.shape)
    
    # tokenize the target using the bart tokenizer
    target_input_ids, target_attention_mask = tokenize(target, DS_tokenizer, max_length = args.summary_length)
    print("Target shape: ")
    print(target_input_ids.shape, target_attention_mask.shape)

    # turn to the tensordataset
    data = TensorDataset(article_input_ids, article_attention_mask, target_input_ids, target_attention_mask)

    return data


# ##################################################################################################################################################
# # Training datset for DS  
train_data = get_dataset(dataset, args.train_num, tokenizer, mode = 'train')# Create the DataLoader for our training set.
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                        batch_size=args.batch_size, pin_memory=True, num_workers=0)

# # Test datset for DS  
test_data = get_dataset(dataset, -1, tokenizer, mode = 'test')# Create the DataLoader for our training set.
test_dataloader = DataLoader(test_data, sampler=RandomSampler(test_data), 
                        batch_size=args.batch_size, pin_memory=True, num_workers=0)

# ##################################################################################################################################################


num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)


max_train_steps = args.epochs * num_update_steps_per_epoch

if args.optimizer == 'adamw':
    logging.info("Loading AdamW optimizer ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

if args.optimizer == 'sgd':
    logging.info("Loading SGD optimizer ...")
    optimizer = torch.optim.SGD(model.parameters(),args.lr,momentum=args.momentum,weight_decay=args.weight_decay)


lr_scheduler = get_scheduler(
        name = args.lr_scheduler_type,
        optimizer= optimizer,
        num_warmup_steps= args.num_warmup_steps,
        num_training_steps= max_train_steps,
    )

completed_steps = 0

loss_list = []

for eps in range(args.epochs):
    # For each batch of training data...
    logging.info('Epoch: %d'% (eps))
    
    epoch_loss = 0

    batch_count = 0
    
    for step, batch in enumerate(train_dataloader):
        
        # push the batch to the cuda
        batch[0] = batch[0].cuda()
        batch[1] = batch[1].cuda()
        batch[2] = batch[2].cuda()
        batch[3] = batch[3].cuda()
        
        model.train()
        
        out = model(input_ids=batch[0], attention_mask =batch[1], labels=batch[2], decoder_attention_mask=batch[3], return_dict=True)
        
        loss = loss_fn(out.logits, batch[2])

        loss = loss / args.gradient_accumulation_steps

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        epoch_loss = epoch_loss + loss.item()

        batch_count =  batch_count + 1

        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

            # For each batch of training data... print loss and the batch number
            
            logging.info("Epoch: %d, Step: %d, Loss: %.3f" % (eps, step, epoch_loss/batch_count))

            optimizer.step()

            lr_scheduler.step()

            optimizer.zero_grad()

            completed_steps += 1

        if completed_steps >= max_train_steps:
            break
    
    # For each batch of training data... print loss and the batch number
    logging.info("Epoch: %d, Loss: %.3f" % (eps, epoch_loss))

    loss_list.append(epoch_loss)

    logging.info(str(tokenizer.batch_decode(model.generate(input_ids = batch[0], num_beams=4, length_penalty=2.0, min_length=10, early_stopping = True, 
        do_sample=False, max_length = args.summary_length, no_repeat_ngram_size = 3, decoder_start_token_id=tokenizer.eos_token_id))))

    logging.info(str(tokenizer.batch_decode(batch[2])))

plt.plot(loss_list)
plt_path = os.path.join(args.save, 'loss_curve.png')
plt.savefig(plt_path)
plt.show()

def infer(test_dataloader, DS_model):
    
    generated_summ = []
    
    ref_summ = []

    for step, batch_test in enumerate(test_dataloader):

        with torch.no_grad():

            DS_model.eval()

            #####################################################################################

            # Input and its attentions
            test_article = Variable(batch_test[0], requires_grad=False).cuda()
            test_article_attn = Variable(batch_test[1], requires_grad=False).cuda()

            # Number of datapoints
            n = test_article.size(0)

            ## Label
            # BART
            test_summary = Variable(batch_test[2], requires_grad=False).cuda()
            test_summary_attn = Variable(batch_test[3], requires_grad=False).cuda()
            ######################################################################################

            generated_summ.extend(tokenizer.batch_decode(DS_model.generate(input_ids = test_article, num_beams=4, length_penalty=2.0, min_length=10, early_stopping = True, 
                do_sample=False, max_length = args.summary_length, no_repeat_ngram_size = 3, decoder_start_token_id=tokenizer.eos_token_id)))
            
            ref_summ.extend(tokenizer.batch_decode(test_summary))

    scores = rouge.get_scores(generated_summ, ref_summ, avg=True)
    
    return scores

model_path = os.path.join(args.save, 'BART_weights.pt')

torch.save(model.state_dict(), model_path)

scores = infer(test_dataloader, model)

logging.info(str(scores))
