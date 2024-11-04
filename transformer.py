# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, num_positions)

        # Multiple Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Embedding + Positional Encoding
        x = self.embedding(indices)
        x = self.pos_encoder(x)

        # Pass through each Transformer layer
        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_maps.append(attn_map)

        # Output layer for classification (log probabilities)
        logit = self.fc_out(x)
        log_probs = torch.log_softmax(logit, dim=-1)

        return log_probs, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        
        # Self-attention layers
        self.attn_query = nn.Linear(d_model, d_internal)
        self.attn_key = nn.Linear(d_model, d_internal)
        self.attn_value = nn.Linear(d_model, d_internal)
        
        # Projection layer to bring d_internal back to d_model
        self.attn_projection = nn.Linear(d_internal, d_model)
        
        # Feedforward layers
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, input_vecs):
        # Self-attention mechanism
        Q = self.attn_query(input_vecs)
        K = self.attn_key(input_vecs)
        V = self.attn_value(input_vecs)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_internal)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Project back to d_model for residual connection
        attn_output = self.attn_projection(attn_output)
        
        # Add & Normalize
        x = self.layer_norm1(input_vecs + attn_output)
        
        # Feedforward layer
        ff_output = self.feedforward(x)
        
        # Add & Normalize
        output = self.layer_norm2(x + ff_output)
        
        return output, attn_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

def train_classifier(args, train, dev):
    """
    Trains a Transformer-based classifier.

    :param args: Argument configurations
    :param train: List of training examples (LetterCountingExample objects)
    :param dev: List of development examples (LetterCountingExample objects)
    :return: Trained model
    """
    # Hyperparameters
    vocab_size = 27 
    num_positions = 20 #################
    d_model = 64  
    d_internal = 32 
    num_classes = 3 
    num_layers = 2  #####################
    learning_rate = 0.001
    num_epochs = 10

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for example in train:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            log_probs, _ = model(example.input_tensor)
            # Reshape log_probs and output_tensor for the criterion
            loss = criterion(log_probs.view(-1, num_classes), example.output_tensor.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Print average training loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train):.4f}")

        # Evaluation on the dev set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for example in dev:
                log_probs, _ = model(example.input_tensor)
                predictions = torch.argmax(log_probs, dim=1)
                correct += (predictions == example.output_tensor).sum().item()
                total += example.output_tensor.size(0)

        accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Dev Accuracy: {accuracy:.2f}%")

    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                # ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                # ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_xticks(np.arange(len(ex.input)))
                ax.set_xticklabels(list(ex.input))  
                ax.set_yticks(np.arange(len(ex.input)))
                ax.set_yticklabels(list(ex.input)) 
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
