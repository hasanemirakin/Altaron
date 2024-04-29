import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

def expand_call_fe(jobs):
    func = jobs["func"]
    args = jobs["args"]
    args["df"] = jobs["data"]

    return func(**args)

def combine_outputs_concat_df(outputs):
    
    out = pd.DataFrame()

    for out_ in outputs:
        out = pd.concat((out, out_))

    return out[~out.index.duplicated()]

def expand_call_parallel_tickers(jobs):
    func = jobs["func"]
    args = jobs["args"]
    tickers = jobs["data"]
    
    if len(tickers) == 1:
        args["ticker"] = tickers[0]
        return {tickers[0]: func(**args)}
    else:
        #multiple tickers in one job
        outputs = {}
        for t in tickers:
            a = args.copy()
            a["ticker"] = t

            outputs[t] = func(**a)
        
        return outputs

def combine_outputs_update_dictionary(outputs):

    out = {}

    for out_ in outputs:
        out.update(out_)
    
    return out

def expand_call_by_dates(jobs):

    func = jobs["func"]
    args = jobs["args"]
    
    dates = jobs["data"]

    outputs = {}

    if len(dates) == 1:
        args["date"] = dates[0]

        return {dates[0]: func(**args)}

    else:
        #multiple dates in one job
        outputs = {}

        for date in dates:
            a = args.copy()
            a["date"] = date

            outputs[date] = func(**a)

        return outputs

def process_jobs(
        jobs,
        call_expansion=expand_call_fe,
        output_combination=combine_outputs_concat_df,
        num_threads=None
):
    
    if num_threads is None:
        num_threads = min(mp.cpu_count(), len(jobs))
    
    pool = mp.Pool(processes=min(num_threads, len(jobs)))    
    out = []

    outputs = pool.map(call_expansion, jobs)

    for out_ in outputs:
        out.append(out_)

    pool.close()
    pool.join()

    return output_combination(out)