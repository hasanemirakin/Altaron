import numpy as np
import multiprocessing as mp

def lin_split(
        n_atoms,
        n_threads
):
    #passing dtype=np.int32 assures integer indexes
    #also functions the same as np.ceil()
    return np.linspace(0, n_atoms, n_threads+1, dtype=np.int32)

def nested_split(
        n_atoms,
        n_threads,
        upper_triangular=False
):

    parts = [0]

    for n in range(n_threads):

        delta = 1 + 4*(parts[-1]**2 + parts[-1] + n_atoms*(n_atoms+1)/n_threads)
        root = (-1 + np.sqrt(delta))/2

        parts.append(root)

    if upper_triangular: #first rows are heaviest 
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.concatenate((np.array([0]), parts))
    
    return np.int32(parts)  

def prepare_jobs(
        func,
        data,
        args,
        num_threads=None,
        linear_split=True,
        **kwargs
):

    if num_threads is None:
        num_threads = mp.cpu_count()
    else:
        num_threads = min(mp.cpu_count(), num_threads)

    n_atoms = len(data)
    n_threads = min(num_threads, n_atoms) 

    if linear_split:
        parts = lin_split(n_atoms=n_atoms, n_threads=n_threads)
    else:
        ut = kwargs.get("upper_triangular", False)
        parts = nested_split(n_atoms=n_atoms, n_threads=n_threads, upper_triangular=ut)
    
    #part extension is to handle nan values arising when applying functions in parralel
    #Ex: an SMA function with window 20 would yield nan values in first 19 indexes normally
    #Applied in parallel, it would yield nan values in first 19 indexes of each separate parts
    #Extending parts by 19 on the left end will allow us to input actual values when combining outputs
    part_extension = kwargs.get("extend_parts", 0)

    jobs = [
        {
            "func": func,
            "data": data[
                max(0,parts[i-1]-part_extension):parts[i]
                ],
            "args": args
        }
    for i in range(1,len(parts))
    ]

    return jobs

def infer_nan_window(
            func,
            args,
            data
    ):
        
        a = args.copy()
        a["df"] = data.copy()
        out = func(**a)

        max_na = out.isna().sum().max()

        return max_na