# Symmetry Transformer

Instead of only predicting the next token also predict the current and previous tokens using separate heads, the notion being learning to predict a part of data input x from any other part of x is beneficial to learning.

## Results

Baseline: ```step 2000: train loss 1.7603, val loss 1.9151```

Symmetry: ```step 2000: train loss 1.6470, val loss 1.8288```

<img width="640" height="480" alt="symmetry_plot" src="https://github.com/user-attachments/assets/53081bde-0507-46db-a3fb-580f13c956f3" />

## Usage

```
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
replace train.py and model.py with the ones in this repo
python3.10 train.py
```

## Key Code

```py
            logits = self.lm_head(x)
            aux = self.lm_head2(x)
            aux2 = self.lm_head3(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if use_aux:
                loss += F.cross_entropy(aux.view(-1, aux.size(-1)), targets2.view(-1), ignore_index=-1)
                loss += F.cross_entropy(aux2.view(-1, aux2.size(-1)), targets3.view(-1), ignore_index=-1)
```
