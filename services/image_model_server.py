import os
import uvicorn
from pydantic import BaseModel 
from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import time

import torch
import torch.multiprocessing as mp
from collections import defaultdict

from contextlib import asynccontextmanager
# look into the import thing. i know __init__.py has to be there.. might be missing something. for now import from same dir
from fineTuneResNet import loadResNet50
# rememebr - here we just want to return the point list in the api response
from refactor_analyze_img import get_subimg_inds, patch_generator, make_padded_image, model_runner

# could inherit from class model_runner. but there's some extra changes i want to make 

'''
need to add in a request blocker till the processes are up, and then a blocker while the models are running, like a queue

try out the service 
curl -X POST http://127.0.0.1:8002/infer \
    -H "Content-Type: application/json" \
    -d '{"img_path": "/mnt/g/data/archive/user_meta_data/patch_sets/bbox_check/patch_0.png"}'
'''

models = {}
processes = 4

# Initialize function for each worker process -- each process has its own memory space. 
# so these are global to that process
def init_worker():
    global worker_model
    worker_model = model_runner()
    print(f"Worker {mp.current_process().name} initialized model")

# Modified worker - uses the global model in each process
def worker(inds_chunk, padimg):
    all_out = defaultdict(float)
    for (y0, y1, x0, x1) in inds_chunk:
        view = padimg[y0:y1, x0:x1, :]
        out = worker_model.process_img(view, offset_y=y0, offset_x=x0, batch_size=32)
        for k_,v_ in out.items():
            all_out[k_]+=v_
    return all_out

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Creating persistent worker pool...")
    mp.set_start_method("fork", force=True)
    # Create pool once with initializer
    models["pool"] = mp.Pool(processes=processes, initializer=init_worker)
    print(f"Pool created with {processes} workers")
    yield
    print("Shutting down pool...")
    models["pool"].close()
    models["pool"].join()

app = FastAPI(lifespan=lifespan)

@app.post("/infer")
def infer(req: dict):
    start_time = time.time()
    img_path = req["img_path"]
    img0 = Image.open(img_path).convert("RGB")
    img0 = np.array(img0) / 255.0
    
    img_size = img0.shape[0]
    inds, pad_up = get_subimg_inds(img_size=img_size, stride=16)

    chunks = np.array_split(inds, processes)
    padimg0 = make_padded_image(img0, [img_size + pad_up, img_size + pad_up, 3])

    # Use the persistent pool - models already loaded in each worker
    results = models["pool"].starmap(worker, [(chunks[i], padimg0) for i in range(processes)])
    
    accumulated = defaultdict(float)
    for rr in results:
        for k_, v_ in rr.items():
            accumulated[k_] += v_

    # Convert tuple keys to lists or strings for JSON serialization
    points_list = [{"yx": list(k), "score": float(v)} for k, v in accumulated.items()]
    elapsed_time = time.time() - start_time
    print(f'Inference took {elapsed_time:.3f} seconds, found {len(accumulated)} points')
    
    return {"points": points_list, "inference_time": elapsed_time}

if __name__ == "__main__":
    # things to do before running the application. 
    uvicorn.run(app, host="127.0.0.1", port=8002)
    