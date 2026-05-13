import os
import math
import random
import time
import sys
import datetime

random.seed(42)

LOG_FILE = 'model_output.log'

class Tee:
    """Mirrors all print() output to a log file. Windows CP1252 safe."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', buffering=1, encoding='utf-8')
        header = f"=== TinyGPT Audited Run | {datetime.datetime.now().isoformat()} ===\n"
        self.log.write(header)

    def write(self, message):
        # Write to terminal safely — encode to CP1252, replace unmappable chars
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            safe = message.encode('cp1252', errors='replace').decode('cp1252')
            self.terminal.write(safe)
        # Log file always gets full UTF-8
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

sys.stdout = Tee(LOG_FILE)
print("Output logging active -> model_output.log")

# ─────────────────────────────────────────────
# STAGE 1: DATA COLLECTION  (Audit.md §2)
# ─────────────────────────────────────────────
t_data_start = time.perf_counter()

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    print("Downloading dataset...")
    urllib.request.urlretrieve(names_url, 'input.txt')
    print("Download complete.")

# Basic data quality check — Audit.md §2 (Data Quality)
raw_lines = open('input.txt').readlines()
docs_raw  = [line.strip() for line in raw_lines if line.strip()]
duplicates = len(docs_raw) - len(set(docs_raw))
empty      = sum(1 for line in raw_lines if not line.strip())

print(f"\n{'='*50}")
print("STAGE 1 — DATA COLLECTION")
print(f"{'='*50}")
print(f"  Total documents : {len(docs_raw):,}")
print(f"  Duplicate names : {duplicates}")
print(f"  Empty lines     : {empty}")
print(f"  Min name length : {min(len(d) for d in docs_raw)}")
print(f"  Max name length : {max(len(d) for d in docs_raw)}")
print(f"  Avg name length : {sum(len(d) for d in docs_raw)/len(docs_raw):.2f}")

t_data_end = time.perf_counter()
print(f"  [TIME]  Data collection time : {t_data_end - t_data_start:.3f}s")

# ─────────────────────────────────────────────
# STAGE 2: TRAIN / VAL / TEST SPLIT  (Audit.md §4 — CRITICAL)
# Deduplication FIRST, then split BEFORE any preprocessing or tokenizer fitting.
# ─────────────────────────────────────────────
t_split_start = time.perf_counter()

# Audit.md §2 fix: deduplicate before splitting so the same name
# cannot appear in both train and val/test sets.
docs_deduped = list(dict.fromkeys(docs_raw))  # preserves order, removes duplicates
n_removed = len(docs_raw) - len(docs_deduped)
print(f"  Deduplication: removed {n_removed} duplicate names ({len(docs_deduped):,} unique remain)")

docs = docs_deduped
random.shuffle(docs)           # shuffle with seeded RNG for reproducibility

n_total = len(docs)
n_train = int(0.80 * n_total)
n_val   = int(0.10 * n_total)
# remaining goes to test

train_docs = docs[:n_train]
val_docs   = docs[n_train : n_train + n_val]
test_docs  = docs[n_train + n_val :]

# Verify no overlap — Audit.md §4 (Split Sanctity)
train_set = set(train_docs)
val_set   = set(val_docs)
test_set  = set(test_docs)
train_val_overlap  = len(train_set & val_set)
train_test_overlap = len(train_set & test_set)
val_test_overlap   = len(val_set   & test_set)

print(f"\n{'='*50}")
print("STAGE 2 — TRAIN / VAL / TEST SPLIT")
print(f"{'='*50}")
print(f"  Train docs : {len(train_docs):,}  ({100*len(train_docs)/n_total:.1f}%)")
print(f"  Val   docs : {len(val_docs):,}   ({100*len(val_docs)/n_total:.1f}%)")
print(f"  Test  docs : {len(test_docs):,}   ({100*len(test_docs)/n_total:.1f}%)")
print(f"  Train/Val  overlap : {train_val_overlap}  (must be 0)")
print(f"  Train/Test overlap : {train_test_overlap}  (must be 0)")
print(f"  Val/Test   overlap : {val_test_overlap}  (must be 0)")

assert train_val_overlap == 0,  "AUDIT FAIL: Train/Val overlap detected!"
assert train_test_overlap == 0, "AUDIT FAIL: Train/Test overlap detected!"
assert val_test_overlap == 0,   "AUDIT FAIL: Val/Test overlap detected!"
print("  [OK] Split sanctity verified — zero overlap")

t_split_end = time.perf_counter()
print(f"  [TIME]  Split time : {t_split_end - t_split_start:.4f}s")

# ─────────────────────────────────────────────
# STAGE 3: TOKENIZER  (fit on TRAIN only — Audit.md §6)
# ─────────────────────────────────────────────

# IMPORTANT: vocab is derived from train_docs only, not all docs
uchars     = sorted(set(''.join(train_docs)))
BOS        = len(uchars)
vocab_size = len(uchars) + 1

print(f"\n{'='*50}")
print("STAGE 3 — TOKENIZER")
print(f"{'='*50}")
print(f"  Vocab size : {vocab_size} (fit on train only)")
print(f"  Characters : {''.join(uchars)}")
print(f"  BOS token  : {BOS}")

# Check for OOV in val/test — Audit.md §6 (OOV risk)
val_chars  = set(''.join(val_docs))
test_chars = set(''.join(test_docs))
train_chars = set(uchars)
oov_val  = val_chars  - train_chars
oov_test = test_chars - train_chars
print(f"  OOV chars in val  : {oov_val  or 'none'}")
print(f"  OOV chars in test : {oov_test or 'none'}")

# ─────────────────────────────────────────────
# STAGE 4: AUTOGRAD ENGINE  (unchanged from original)
# ─────────────────────────────────────────────

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data        = data
        self.grad        = 0
        self._children   = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):            return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self):            return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self):           return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self):        return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

print("[OK] Autograd engine defined.")

# ─────────────────────────────────────────────
# STAGE 5: MODEL DEFINITION  (Audit.md §7)
# ─────────────────────────────────────────────
t_model_start = time.perf_counter()

n_layer    = 1
n_embd     = 16
block_size = 16
n_head     = 4
head_dim   = n_embd // n_head

matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
]

state_dict = {
    'wte':     matrix(vocab_size, n_embd),
    'wpe':     matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps    = [(val - max_val).exp() for val in logits]
    total   = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms    = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs  = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits  = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                            for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out     = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                            for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

t_model_end = time.perf_counter()

print(f"\n{'='*50}")
print("STAGE 5 — MODEL DEFINITION")
print(f"{'='*50}")
print(f"  Architecture   : GPT (n_layer={n_layer}, n_embd={n_embd}, n_head={n_head})")
print(f"  Num parameters : {len(params):,}")
print(f"  Block size     : {block_size}")
print(f"  Vocab size     : {vocab_size}")
print(f"  [TIME]  Model init time : {t_model_end - t_model_start:.3f}s")

# ─────────────────────────────────────────────
# EVALUATION HELPER — computes average loss on a doc list
# ─────────────────────────────────────────────

def compute_loss(doc_list, max_docs=300):
    """Compute average cross-entropy loss over a sample of documents."""
    sample = doc_list[:max_docs]
    total_loss = 0.0
    total_n    = 0
    for doc in sample:
        # Skip docs with chars unseen in training vocab (OOV safety)
        if any(ch not in train_chars for ch in doc):
            continue
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n      = min(block_size, len(tokens) - 1)
        keys_e   = [[] for _ in range(n_layer)]
        values_e = [[] for _ in range(n_layer)]
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys_e, values_e)
            probs  = softmax(logits)
            total_loss += (-probs[target_id].log()).data
            total_n    += 1
    return total_loss / total_n if total_n > 0 else float('nan')

print("[OK] Evaluation helper defined.")

# ─────────────────────────────────────────────
# STAGE 6: TRAINING  (Audit.md §8)
# ─────────────────────────────────────────────
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)

num_steps    = 1000
val_interval = 100    # evaluate on val set every N steps

train_losses = []   # (step, loss) for audit trail
val_losses   = []   # (step, loss) for audit trail

t_train_start = time.perf_counter()

print(f"\n{'='*50}")
print("STAGE 6 — TRAINING")
print(f"{'='*50}")
print(f"  Steps        : {num_steps}")
print(f"  Val interval : every {val_interval} steps")
print(f"  LR           : {learning_rate} (linear decay)")
print()

for step in range(num_steps):
    doc    = train_docs[step % len(train_docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n      = min(block_size, len(tokens) - 1)

    keys_t   = [[] for _ in range(n_layer)]
    values_t = [[] for _ in range(n_layer)]
    losses_t = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits   = gpt(token_id, pos_id, keys_t, values_t)
        probs    = softmax(logits)
        loss_tok = -probs[target_id].log()
        losses_t.append(loss_tok)

    loss = (1 / n) * sum(losses_t)
    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat    = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat    = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data  -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad   = 0

    train_losses.append((step + 1, loss.data))

    # Periodic validation — Audit.md §8 (Overfitting monitoring)
    if (step + 1) % val_interval == 0:
        t_val_start = time.perf_counter()
        val_loss    = compute_loss(val_docs, max_docs=200)
        t_val_end   = time.perf_counter()
        val_losses.append((step + 1, val_loss))
        gap = val_loss - loss.data
        elapsed = time.perf_counter() - t_train_start
        print(f"  step {step+1:4d}/{num_steps} | "
              f"train_loss={loss.data:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"gap={gap:+.4f} | "
              f"elapsed={elapsed:.1f}s | "
              f"val_eval={t_val_end-t_val_start:.2f}s")

t_train_end = time.perf_counter()
total_train_time = t_train_end - t_train_start
print(f"\n  [TIME]  Total training time : {total_train_time:.1f}s  "
      f"({total_train_time/num_steps*1000:.1f}ms/step)")

# ─────────────────────────────────────────────
# STAGE 7: MODEL EVALUATION — held-out TEST SET
# Evaluated EXACTLY ONCE — Audit.md §10 (Test Set Sanctity)
# ─────────────────────────────────────────────
t_eval_start = time.perf_counter()

final_train_loss = compute_loss(train_docs, max_docs=300)
final_val_loss   = compute_loss(val_docs,   max_docs=300)
final_test_loss  = compute_loss(test_docs,  max_docs=300)  # ← evaluated ONCE, here, never before

t_eval_end = time.perf_counter()

gen_gap_val  = final_val_loss  - final_train_loss
gen_gap_test = final_test_loss - final_train_loss

print(f"\n{'='*50}")
print("STAGE 7 — MODEL EVALUATION REPORT")
print(f"{'='*50}")
print(f"  Final train loss : {final_train_loss:.4f}")
print(f"  Final val   loss : {final_val_loss:.4f}   (gap = {gen_gap_val:+.4f})")
print(f"  Final test  loss : {final_test_loss:.4f}   (gap = {gen_gap_test:+.4f})")
print()

# Generalisation assessment
if gen_gap_test < 0.05:
    verdict = "[OK] GOOD — model generalises well"
elif gen_gap_test < 0.20:
    verdict = "[WARN]  MODERATE — mild overfitting, consider regularisation"
else:
    verdict = "[FAIL] HIGH — significant overfitting detected"
print(f"  Generalisation verdict : {verdict}")
print(f"  [TIME]  Evaluation time : {t_eval_end - t_eval_start:.2f}s")

# Training curve summary
print(f"\n  Training loss curve (sampled):")
for step, tl in train_losses[::100]:
    print(f"    step {step:4d} | train_loss = {tl:.4f}")

print(f"\n  Validation loss curve:")
for step, vl in val_losses:
    print(f"    step {step:4d} | val_loss   = {vl:.4f}")

# ─────────────────────────────────────────────
# STAGE 8: INFERENCE + MEMORISATION CHECK
# ─────────────────────────────────────────────
t_infer_start = time.perf_counter()

temperature   = 0.5
n_samples     = 20
generated     = []

for sample_idx in range(n_samples):
    keys_i   = [[] for _ in range(n_layer)]
    values_i = [[] for _ in range(n_layer)]
    token_id = BOS
    sample   = []
    for pos_id in range(block_size):
        logits   = gpt(token_id, pos_id, keys_i, values_i)
        probs    = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    generated.append(''.join(sample))

t_infer_end = time.perf_counter()

# Memorisation check — Audit.md §10 (Leakage / Memorisation)
memorised = [g for g in generated if g in train_set]
novel     = [g for g in generated if g not in train_set]

print(f"\n{'='*50}")
print("STAGE 8 — INFERENCE OUTPUT")
print(f"{'='*50}")
print(f"  Temperature : {temperature}")
print(f"  Samples     : {n_samples}")
print()
for i, name in enumerate(generated, 1):
    flag = "[MEMORISED]" if name in train_set else ""
    print(f"  sample {i:2d}: {name:<20} {flag}")

print(f"\n  Memorisation rate : {len(memorised)}/{n_samples} samples "
      f"({100*len(memorised)/n_samples:.0f}%)")
print(f"  Novel outputs     : {len(novel)}/{n_samples}")
print(f"  [TIME]  Inference time : {t_infer_end - t_infer_start:.3f}s")

# ─────────────────────────────────────────────
# FINAL SUMMARY REPORT
# ─────────────────────────────────────────────
total_wall_time = time.perf_counter() - t_data_start

print(f"\n{'='*50}")
print("AUDIT SUMMARY — TinyGPT")
print(f"{'='*50}")
print(f"  Timestamp         : {datetime.datetime.now().isoformat()}")
print(f"  Model params      : {len(params):,}")
print(f"  Train docs        : {len(train_docs):,}")
print(f"  Val docs          : {len(val_docs):,}")
print(f"  Test docs         : {len(test_docs):,}")
print(f"  Train loss (final): {final_train_loss:.4f}")
print(f"  Val   loss (final): {final_val_loss:.4f}")
print(f"  Test  loss (final): {final_test_loss:.4f}")
print(f"  Generalisation    : {verdict}")
print(f"  Memorisation rate : {100*len(memorised)/n_samples:.0f}%")
print(f"  Total wall time   : {total_wall_time:.1f}s")
print(f"  Log file          : {LOG_FILE}")
print(f"{'='*50}")

# Close the logger
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
print(f"\nRun complete. Full output saved to -> {LOG_FILE}")