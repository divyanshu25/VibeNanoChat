# Async Prefetching: Stop Making Your GPU Wait for Data

## tl;dr

Your GPU was sitting idle every few steps waiting for the CPU to read from disk and tokenize documents. We fixed it by moving all that slow stuff to a background thread. MFU went from bouncing between 35-47% to a steady 46-47%. Simple fix, big impact.

## The Problem

Look at these training logs:

```
[Step  1164] Speed: 1911.1K tok/s | MFU: 46.76% | Time: 0.27s
[Step  1165] Speed: 1514.8K tok/s | MFU: 37.06% | Time: 0.35s  ← WTF?
[Step  1166] Speed: 1847.4K tok/s | MFU: 45.20% | Time: 0.28s
[Step  1167] Speed: 1895.5K tok/s | MFU: 46.38% | Time: 0.28s
[Step  1168] Speed: 1513.3K tok/s | MFU: 37.03% | Time: 0.35s  ← There it is again
```

See that? Every few steps we get a 25% slowdown. MFU drops from 46% to 37%. Not great.

## Why This Happens

The dataloader keeps a buffer of tokenized documents. When it runs low, it refills by:
1. Reading a chunk from the Parquet file on disk (~20-30ms)
2. Tokenizing 128 documents (~30-50ms)

This happens **in the training loop**. So the training step looks like:

```
Step N (normal):
  - Forward pass
  - Backward pass
  - Optimizer step
  Time: 280ms, GPU busy the whole time

Step N+1 (buffer refill):
  - Read from disk      [GPU idle 20ms]
  - Tokenize documents  [GPU idle 50ms]
  - Forward pass
  - Backward pass
  - Optimizer step
  Time: 350ms, GPU wasted 70ms waiting

Step N+2 (normal again):
  - Forward pass
  - Backward pass
  - Optimizer step
  Time: 280ms
```

That 70ms of GPU idle time is the problem. It's like having a Ferrari but stopping every few miles to refuel instead of keeping the tank full.

## The Fix: Background Prefetching

The idea is simple: keep a background thread running that continuously:
1. Reads from disk
2. Tokenizes documents
3. Puts them in a queue

The training loop just grabs from the queue, which is instant. The queue acts as a buffer between the slow I/O and the fast training.

Here's the architecture:

```
Background Thread (runs continuously):
  while True:
    text_batch = read_from_parquet()        # Slow, but who cares, we're not blocking training
    tokenized = tokenize(text_batch)        # Also slow, also fine
    queue.put(tokenized)                    # Instant (unless queue is full, which is good)

Training Loop:
  while training:
    docs = queue.get(timeout=0.001)         # Instant! Already tokenized!
    batch = pack_documents(docs)
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

The background thread is always running, always filling the queue. By the time you need more documents, they're already sitting there ready to go.

## Implementation

Here's what changed in the dataloader:

### 1. Added a queue and background thread

```python
import queue
import threading

# In __init__:
self.prefetch_queue = queue.Queue(maxsize=4)  # Hold 4 batches of tokenized docs
self.prefetch_stop = threading.Event()         # Signal to stop on shutdown
self.prefetch_thread = threading.Thread(
    target=self._prefetch_worker,
    daemon=True,
    name=f"Prefetch-{split}-rank{ddp_rank}"
)
self.prefetch_thread.start()
```

Why `maxsize=4`? Each batch is ~128 tokenized documents. So we're keeping ~512 documents (2-4 training batches worth) ready to go. That's enough to absorb any timing variance without using too much memory (~1-2MB).

### 2. The prefetch worker

```python
def _prefetch_worker(self):
    """Background thread that does all the slow stuff."""
    try:
        while not self.prefetch_stop.is_set():
            # Fetch raw text from Parquet (SLOW - disk I/O)
            text_batch, epoch = next(self.batches)
            
            # Tokenize with BOS (SLOW - CPU intensive)
            tokenized_documents = []
            for document_text in text_batch:
                token_ids = self.tokenizer.encode(
                    f"<|bos|>{document_text}", allowed_special="all"
                )
                if len(token_ids) > 0:
                    tokenized_documents.append(token_ids)
            
            # Put in queue (FAST - unless queue is full, then we wait)
            if not self.prefetch_stop.is_set():
                self.prefetch_queue.put((tokenized_documents, epoch))
                
    except Exception as e:
        if not self.prefetch_stop.is_set():
            print(f"⚠️  Prefetch worker crashed: {e}")
```

This runs forever in the background. If the queue fills up, `.put()` blocks, which is exactly what we want - natural backpressure to match the training speed.

### 3. Modified buffer refill

```python
def _refill_buffer(self, initial=False):
    """Now just pulls from queue instead of doing I/O."""
    target_size = self.buffer_size if initial else self.buffer_size // 2
    docs_to_add = target_size - len(self.doc_buffer)
    
    if docs_to_add <= 0:
        return
    
    documents_added = 0
    while documents_added < docs_to_add:
        try:
            # For initial fill: block until data is ready
            # During training: non-blocking, return immediately if empty
            timeout = 5.0 if initial else 0.001
            tokenized_documents, epoch = self.prefetch_queue.get(timeout=timeout)
            self.current_epoch = epoch
            
            for token_ids in tokenized_documents:
                self.doc_buffer.append(token_ids)
                documents_added += 1
                if documents_added >= docs_to_add:
                    break
                    
        except queue.Empty:
            if initial:
                continue  # Keep waiting during startup
            else:
                break  # Don't block training
```

The key change: instead of blocking on `next(self.batches)` and `tokenizer.encode()`, we just grab from the queue with a 1ms timeout. If the queue is empty (shouldn't happen), we just use what we have and move on.

## Results

Before:
```
Step 1165: MFU: 37.06% | Time: 0.35s  ← Slow
Step 1166: MFU: 45.20% | Time: 0.28s
Step 1167: MFU: 46.38% | Time: 0.28s
Step 1168: MFU: 37.03% | Time: 0.35s  ← Slow
```

After:
```
Step 1165: MFU: 46.52% | Time: 0.28s  ← Smooth
Step 1166: MFU: 46.48% | Time: 0.28s
Step 1167: MFU: 46.55% | Time: 0.27s
Step 1168: MFU: 46.51% | Time: 0.28s  ← Smooth
```

MFU is now consistently 46-47%. No more drops to 37%. The GPU stays busy.

## Monitoring

You can check if prefetching is working by looking at the queue size:

```python
stats = dataloader.get_stats()
print(f"Queue: {stats['prefetch_queue_size']}/4")
```

Ideal: Queue size stays at 4 (full). This means the background thread is keeping up, and you always have data ready.

If you see queue size at 0-1, something is wrong:
- Disk I/O might be slow (network filesystem?)
- Tokenization might be slow (increase `tokenizer_batch_size`?)
- Buffer might be too small (increase `bos_dataloader_buffer_size`?)

## Why This Works

The key insight: **CPU I/O and GPU compute can happen simultaneously**. While the GPU is doing forward/backward passes, the CPU can be reading from disk and tokenizing. By using a background thread, we overlap these operations instead of doing them sequentially.

Old way (sequential):
```
[Read disk] → [Tokenize] → [GPU forward] → [GPU backward] → [Read disk] → ...
     ↑            ↑
   GPU idle     GPU idle
```

New way (overlapped):
```
Background thread: [Read disk] → [Tokenize] → [Read disk] → [Tokenize] → ...
Training loop:                   [GPU forward] → [GPU backward] → [GPU forward] → ...
```

The GPU never waits. The training loop never waits. Everyone's happy.

## Some Details

**Thread safety**: Python's `queue.Queue` is thread-safe, so we don't need locks. The background thread writes to the queue, the main thread reads from it, no shared state mutation.

**Memory**: Each batch in the queue is ~128 tokenized documents, maybe 0.5MB each. Total overhead: ~2MB. Negligible.

**CPU usage**: Background thread uses ~10-20% of one CPU core for I/O and tokenization. Modern servers have 32-128 cores, so this is fine.

**Shutdown**: The thread is marked as `daemon=True`, so it automatically dies when the program exits. We also have `prefetch_stop.is_set()` checks for clean shutdown if needed.

**What if it crashes?**: If the background thread crashes, we log the error and training continues. The queue will drain, and `_refill_buffer()` will just return immediately (timeout), so you'll use whatever documents are left in the buffer. Training won't crash, but you might see jitter again until you restart.

## Configuration

Default settings work well:

```python
bos_dataloader_buffer_size = 1000    # Main document buffer
tokenizer_batch_size = 128           # Docs per tokenization batch
prefetch_queue_maxsize = 4           # Batches in queue
```

If training is very fast (small models), you might want more prefetch:
```python
prefetch_queue_maxsize = 8
```

If you're on slow storage (network FS, HDD), increase everything:
```python
bos_dataloader_buffer_size = 4000
prefetch_queue_maxsize = 8
tokenizer_batch_size = 256
```

## Future Improvements

**Batch tokenization**: Right now we tokenize documents one by one. The tokenizer can actually do batches:

```python
# Current (sequential):
for doc in text_batch:
    tokens = tokenizer.encode(f"<|bos|>{doc}")

# Potential (batched):
prefixed = [f"<|bos|>{doc}" for doc in text_batch]
tokens = tokenizer(prefixed)  # 2-3x faster
```

This would make the prefetch thread even faster, allowing smaller queue sizes or handling slower disks better.

**Memory-mapped Parquet**: PyArrow can memory-map Parquet files instead of reading them into memory. This can reduce latency by ~30% for hot data.

**Multiple prefetch threads**: If you have really fast training or really slow storage, you could run 2-3 prefetch threads feeding the same queue. Diminishing returns though - usually one thread is enough.

## Bottom Line

Async prefetching is a simple idea: don't make the GPU wait for the CPU. Keep data flowing through a queue, let the background thread handle the slow I/O, and the training loop just grabs ready-to-use data.

Result: Smooth 46-47% MFU instead of 35-47% jitter. ~25% improvement in worst-case throughput. GPU stays busy. Training goes brrr.

---

Implementation: `src/dataloaders/fineweb_edu_parquet_bos_dataloader.py`
