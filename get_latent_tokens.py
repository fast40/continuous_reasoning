import os
import yaml
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import tqdm

from coconut.coconut import Coconut
from coconut.dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from coconut.utils import Config, set_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

with open('coconut/args/gsm_coconut_eval.yaml') as f:
    config_dict = yaml.safe_load(f)

configs = Config(config_dict)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens("<|start-latent|>")
tokenizer.add_tokens("<|end-latent|>")
tokenizer.add_tokens("<|latent|>")

latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

model = AutoModelForCausalLM.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model = Coconut(
    model,
    latent_id,
    start_id,
    end_id,
    tokenizer.eos_token_id,
)
model.eval()
model.to(DEVICE)

for i in [4, 25]: #range(4, 26):
    t = time.perf_counter()
    ckpt_dir = snapshot_download(f'Onlydrinkwater/gpt2-coconut-checkpoint{i}')
    state_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    dataset = get_question_latent_dataset(
        i - 1,
        get_dataset('coconut/data/gsm_valid.json', tokenizer),
        configs,
        start_id,
        latent_id,
        end_id,
    )

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    latent_vecs = torch.empty((0, 768))  # (num_latents, hidden)

    num_batches = int(2**13 / BATCH_SIZE)

    for batch_num in tqdm.tqdm(range(num_batches)):
        batch = next(iter(loader))

        input_ids = batch['input_ids'].to(DEVICE)
        labels = input_ids.clone()  # this is UNUSED I think. TODO: figure out if that's actually true
        attention_mask = batch['attention_mask'].to(DEVICE)
        position_ids = batch['position_ids'].to(DEVICE)

        with torch.no_grad():
            out = model.forward(input_ids, attention_mask, labels, position_ids)

        latent_pos = (input_ids == latent_id).nonzero(as_tuple=False) # latent_pos is (num_latents_total_in_batch, 2) => [batch_idx, seq_idx]
        latent_vecs = torch.cat((latent_vecs, out.inputs_embeds[latent_pos[:, 0], latent_pos[:, 1], :].cpu()))

        if batch_num == num_batches - 1:
            torch.save(latent_vecs, f'latent_vecs_48k_checkpoint_{i}.pt')
            t = time.perf_counter() - t
            print(f'checkpoint {i} done in {t} seconds')

