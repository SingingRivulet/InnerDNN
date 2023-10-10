import argparse
import struct
import torch
from typing import Dict
import sys

model = torch.load(sys.argv[1], map_location='cpu')
numLayer = 0
dim = None
dim_hidden = None
for k in model.keys():
    tensor = model[k].float()
    name = k.split(".")
    if name[0] == "blocks":
        numLayer = max(int(name[1])+1, numLayer)
        if name[2] == "ffn" and name[3] == "key":
            if dim_hidden == None:
                dim_hidden = tensor.shape[0]
            if dim == None:
                dim = tensor.shape[1]
            if dim_hidden != tensor.shape[0]:
                print("error:dim_hidden != tensor.shape[0]")
            if dim != tensor.shape[1]:
                print("error:dim != tensor.shape[1]")
    print(k, tensor.shape)
emb_size = model["emb.weight"].shape[0]
out_size = model["head.weight"].shape[0]
print("")
print("emb_size:", emb_size)
print("out_size:", out_size)
print("numLayer:", numLayer)
print("dim:", dim)
print("dim_hidden:", dim_hidden)

with open(sys.argv[2], 'wb') as out_file:
    out_file.write(struct.pack(
        "iiiii",
        dim,
        dim_hidden,
        out_size,
        numLayer,
        emb_size
    ))
    model["emb.weight"].float().numpy().tofile(out_file)
    
    for i in range(numLayer):
        model[f"blocks.{i}.ln1.weight"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.ln1.bias"].float().numpy().tofile(out_file)

    for i in range(numLayer):
        model[f"blocks.{i}.att.time_first"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.att.time_decay"].float().numpy().tofile(out_file)

    for i in range(numLayer):
        model[f"blocks.{i}.att.time_mix_k"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.att.time_mix_v"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.att.time_mix_r"].float().numpy().tofile(out_file)

    for i in range(numLayer):
        model[f"blocks.{i}.att.output.weight"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.att.receptance.weight"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.att.key.weight"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.att.value.weight"].float().numpy().tofile(out_file)

    for i in range(numLayer):
        model[f"blocks.{i}.ffn.time_mix_k"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.ffn.time_mix_r"].float().numpy().tofile(out_file)
        
    for i in range(numLayer):
        model[f"blocks.{i}.ln2.weight"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.ln2.bias"].float().numpy().tofile(out_file)
        
    for i in range(numLayer):
        model[f"blocks.{i}.ffn.receptance.weight"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.ffn.key.weight"].float().numpy().tofile(out_file)
    for i in range(numLayer):
        model[f"blocks.{i}.ffn.value.weight"].float().numpy().tofile(out_file)
        
    model["blocks.0.ln0.weight"].float().numpy().tofile(out_file)
    model["blocks.0.ln0.bias"].float().numpy().tofile(out_file)
    model["ln_out.weight"].float().numpy().tofile(out_file)
    model["ln_out.bias"].float().numpy().tofile(out_file)
    model["head.weight"].float().numpy().tofile(out_file)
