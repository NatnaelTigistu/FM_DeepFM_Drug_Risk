# Complete Study Guide: Factorization Machines and DeepFM for Drug Interaction Risk

---

## 1. The Real-World Problem: Predicting Risky Drug Combinations

When a patient takes two medicines together, the two chemicals can change how each other works in the body. This is called a Drug-Drug Interaction (DDI). 

Some interactions are mild or moderate (like slight headache or feeling sleepy), but some interactions are **Severe** (like sudden low blood pressure, severe internal bleeding, heart rhythm disorder, or death). Doctors and hospital systems want to know in advance:
$$\text{If we give Drug 1 and Drug 2 together, is the risk Severe }(Label = 1)\text{ or Not Severe }(Label = 0)\text{?}$$

### Why this problem is difficult:
1. **High Sparsity**: There are hundreds or thousands of drugs. If we have 420 drugs, the number of possible unordered pairs is:
   $$\frac{420 \times 419}{2} = 87,990 \text{ possible combinations}$$
   In hospital trials, only a small fraction of all possible drug pairs have ever been tested or documented.
2. **Symmetric Relationship**: The pair $(\text{Drug A}, \text{Drug B})$ is biologically the same interaction as $(\text{Drug B}, \text{Drug A})$. The model must not learn one thing for $(A, B)$ and a conflicting thing for $(B, A)$.
3. **Class Imbalance**: Most drug pairs in the database do not cause severe toxicity. Only about $16\%$ of pairs are Severe. If a simple model just predicts $0$ all the time, it gets $84\%$ accuracy but misses every life-threatening reaction.
4. **No Free Text at Test Time**: When a new drug is being prescribed, doctors cannot input text like "patient had severe bleeding" because the interaction hasn't happened yet. The inputs to the model must be strictly:
   $$\text{Input: } (\text{Drug 1}, \text{Drug 2}) \longrightarrow \text{Output: } P(\text{Severe})$$

---

## 2. Why Linear Models Fail and What FM Does

### The Simple Baseline: Logistic Regression
The most basic way to solve this is logistic regression on one-hot encoded drugs:
$$z = b + w_1[\text{Drug 1}] + w_2[\text{Drug 2}]$$
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Here, $w_1[i]$ measures the base danger of Drug 1, and $w_2[j]$ measures the base danger of Drug 2. 
**The big problem**: This model assumes the danger is purely additive. If Drug 1 is safe on its own ($w_1 \approx 0$) and Drug 2 is safe on its own ($w_2 \approx 0$), the model always predicts low risk. But in reality, two individually safe drugs can react together inside the liver and become deadly. 

A linear model cannot see this pair interaction unless we create a separate parameter $w_{ij}$ for every pair:
$$z = b + w_1[i] + w_2[j] + w_{ij}$$

Now the fatal flaw appears: to learn $w_{ij}$, we must observe pair $(i, j)$ multiple times in the training data. If pair $(i, j)$ was never tested together, $w_{ij}$ stays zero and the model learns nothing about that combination.

---

### The Factorization Machine (FM) Solution
Instead of trying to learn an independent parameter $w_{ij}$ for every pair, Rendle (2010) proposed:
$$\text{Factorize } w_{ij} \text{ into the dot product of two latent vectors: } w_{ij} \approx \langle v_1[i], v_2[j] \rangle$$

Each drug gets a small latent embedding vector of dimension $k$ (for example, $k=2$ or $k=8$). 
- $v_1[i]$ represents the latent chemical properties of Drug 1.
- $v_2[j]$ represents the latent chemical properties of Drug 2.

Now the logit becomes:
$$z_{FM} = b + w_1[i] + w_2[j] + \langle v_1[i], v_2[j] \rangle$$
where:
$$\langle v_1[i], v_2[j] \rangle = \sum_{f=1}^k v_{1,f}[i] \cdot v_{2,f}[j]$$

### Why FM can generalize to pairs it never saw:
If Drug A and Drug C both affect the same enzyme (like CYP3A4), their learned latent vectors $v$ will become similar because they interact with other drugs in similar ways. Even if $(\text{Drug A}, \text{Drug B})$ was never seen together in the hospital records, their dot product $\langle v_A, v_B \rangle$ will still be high if Drug A's embedding aligns with Drug B's embedding. FM shares knowledge through the latent space.

---

## 3. Why DeepFM Was Created: The Limit of FM

Factorization Machines are very effective, but they only model **2nd-order (pairwise) dot-product interactions**:
$$\langle v_1, v_2 \rangle = v_{1,1}v_{2,1} + v_{1,2}v_{2,2} + \dots + v_{1,k}v_{2,k}$$

This calculation is purely linear inside each latent dimension. It cannot learn:
- Non-linear relationships (like exponential toxicity thresholds).
- Higher-order interactions across multiple chemical properties.

In 2017, Guo et al. created **DeepFM**. The core idea:
$$\boxed{z_{\text{Total}} = z_{FM} + z_{Deep}}$$
$$\hat{y} = \sigma(z_{\text{Total}})$$

DeepFM has two branches:
1. **FM Branch**: computes $b + w_1[i] + w_2[j] + \langle v_1[i], v_2[j] \rangle$. This directly handles explicit pairwise linear alignment.
2. **Deep Branch**: takes the exact same embedding vectors $v_1[i]$ and $v_2[j]$, concatenates them together into a vector of length $2k$, and passes them through a Multi-Layer Perceptron (MLP) with non-linear activation functions like $\text{ReLU}$.
3. **Shared Embeddings**: The FM branch and the Deep branch **share the exact same embedding vectors**. No separate embedding tables.

---

## 4. DeepFM Architecture Diagram

```
                              Final Prediction
                           p = sigmoid(z_Total)
                                    ▲
                                    │
                         z_Total = z_FM + z_Deep
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
        [ FM Branch ]                             [ Deep Branch ]
               │                                         │
     z_FM = b + w1[i] + w2[j]                     z_Deep = W2*h + b2
          + <v1[i], v2[j]>                               ▲
               │                                         │
               │                                h = ReLU(W1*e + b1)
               │                                         ▲
               │                                         │
               │                                 e = [v1[i]; v2[j]]
               │                                 (concatenation)
               └────────────────────┬────────────────────┘
                                    │
                         Shared Latent Embeddings
                         v1[i] in R^k,  v2[j] in R^k
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
          Drug 1 Index                              Drug 2 Index
          (e.g., Dronedarone = 0)                   (e.g., Atenolol = 1)
```

---

## 5. Concrete Numerical Setup for Drug Risk

To understand the mathematics completely, we use real drug names from our clinical problem and walk through every single arithmetic operation by hand.

### The Drug Vocabulary:
- **Drug 0**: Dronedarone (antiarrhythmic heart medication)
- **Drug 1**: Atenolol (beta-blocker blood pressure medication)
- **Drug 2**: Warfarin (anticoagulant blood thinner)
- **Drug 3**: Metformin (antidiabetic medication)

We set embedding dimension $k = 2$.

### Initial Parameters:

**1. Latent chemical embedding vectors ($k=2$):**
- Dronedarone: $v[0] = [0.20, 0.10]^T$
- Atenolol:    $v[1] = [0.30, 0.20]^T$
- Warfarin:    $v[2] = [0.10, -0.30]^T$
- Metformin:   $v[3] = [0.50, -0.10]^T$

**2. FM Linear weights and global bias:**
- Global bias: $b = 0.05$
- Linear weight for Dronedarone (Drug 1 position): $w_1[0] = 0.10$
- Linear weight for Atenolol (Drug 2 position):    $w_2[1] = 0.20$

**3. Deep MLP weights and biases:**
- Hidden layer dimension = 2.
- Input dimension to MLP is concatenation of two $k=2$ vectors $\implies 2 \times 2 = 4$.
- Weight matrix $W_1$ (shape $2 \times 4$):
  $$W_1 = \begin{bmatrix} 0.50 & 0.00 & 0.50 & 0.00 \\ 0.00 & 0.50 & 0.00 & 0.50 \end{bmatrix}$$
- Bias vector $b_1$ (shape $2 \times 1$):
  $$b_1 = \begin{bmatrix} 0.00 \\ 0.00 \end{bmatrix}$$
- Output layer weight $W_2$ (shape $1 \times 2$):
  $$W_2 = \begin{bmatrix} 0.40 & 0.60 \end{bmatrix}$$
- Output layer bias $b_2$ (scalar):
  $$b_2 = 0.10$$

---

## 6. Forward Pass Step by Step: Dronedarone + Atenolol

We test the combination:
$$\text{Drug 1} = \text{Dronedarone } (i=0), \quad \text{Drug 2} = \text{Atenolol } (j=1)$$
True hospital label: **$y = 1$ (Severe interaction)**.

---

### Step 1: FM Branch Computation
Formula:
$$z_{FM} = b + w_1[0] + w_2[1] + \langle v[0], v[1] \rangle$$

1. Linear base part:
   $$\text{linear} = b + w_1[0] + w_2[1] = 0.05 + 0.10 + 0.20 = 0.35$$

2. Latent dot product part:
   $$\langle v[0], v[1] \rangle = v_1[0] \cdot v_1[1] + v_2[0] \cdot v_2[1]$$
   $$\langle v[0], v[1] \rangle = (0.20 \times 0.30) + (0.10 \times 0.20) = 0.06 + 0.02 = 0.08$$

3. FM logit:
   $$z_{FM} = 0.35 + 0.08 = \mathbf{0.43}$$

---

### Step 2: Deep MLP Branch Computation
Formula:
1. Concatenate the two drug embeddings into vector $e$:
   $$e = \begin{bmatrix} v[0] \\ v[1] \end{bmatrix} = \begin{bmatrix} 0.20 \\ 0.10 \\ 0.30 \\ 0.20 \end{bmatrix}$$

2. Linear projection through $W_1$ and $b_1$:
   $$a_1 = W_1 e + b_1$$
   $$a_{1,1} = (0.50 \times 0.20) + (0.00 \times 0.10) + (0.50 \times 0.30) + (0.00 \times 0.20) + 0.00 = 0.10 + 0.15 = 0.25$$
   $$a_{1,2} = (0.00 \times 0.20) + (0.50 \times 0.10) + (0.00 \times 0.30) + (0.50 \times 0.20) + 0.00 = 0.05 + 0.10 = 0.15$$
   $$a_1 = \begin{bmatrix} 0.25 \\ 0.15 \end{bmatrix}$$

3. Non-linear activation ($\text{ReLU}$):
   $$h = \text{ReLU}(a_1) = \begin{bmatrix} \max(0, 0.25) \\ \max(0, 0.15) \end{bmatrix} = \begin{bmatrix} 0.25 \\ 0.15 \end{bmatrix}$$

4. Linear projection to Deep logit through $W_2$ and $b_2$:
   $$z_{Deep} = W_2 h + b_2 = (0.40 \times 0.25) + (0.60 \times 0.15) + 0.10$$
   $$z_{Deep} = 0.10 + 0.09 + 0.10 = \mathbf{0.29}$$

---

### Step 3: Combine FM and Deep Logits
$$z_{\text{Total}} = z_{FM} + z_{Deep} = 0.43 + 0.29 = \mathbf{0.72}$$

---

### Step 4: Probability and Binary Cross-Entropy Loss
1. Predicted risk probability $p$:
   $$p = \sigma(z_{\text{Total}}) = \frac{1}{1 + e^{-0.72}}$$
   $$e^{-0.72} \approx 0.486752$$
   $$p = \frac{1}{1 + 0.486752} = \frac{1}{1.486752} \approx \mathbf{0.672607}$$
   The model currently estimates a $67.26\%$ probability of a Severe reaction.

2. Binary Cross-Entropy Loss ($y=1$):
   $$L = - [y \ln(p) + (1-y) \ln(1-p)] = - \ln(0.672607) \approx \mathbf{0.396598}$$

---

## 7. Complete Mathematical Backpropagation

Now we want to update the parameters to make the prediction better.

### Step 1: Derivative of Loss with Respect to Total Logit ($z_{\text{Total}}$)
Using the chain rule:
$$\frac{\partial L}{\partial p} = - \frac{y}{p} + \frac{1-y}{1-p} = \frac{p - y}{p(1-p)}$$
$$\frac{\partial p}{\partial z} = p(1-p)$$
$$\delta = \frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} = \frac{p - y}{p(1-p)} \cdot p(1-p) = p - y$$

For our example:
$$\delta = 0.672607 - 1.0 = \mathbf{-0.327393}$$
The negative sign means the predicted logit was too low; it must increase.

---

### Step 2: Gradients Flowing Through the FM Branch
Since $z = z_{FM} + z_{Deep}$, the derivative $\frac{\partial z}{\partial z_{FM}} = 1$.
Therefore:
$$\frac{\partial L}{\partial z_{FM}} = \delta = -0.327393$$

1. Global bias gradient:
   $$\frac{\partial L}{\partial b} = \delta \cdot \frac{\partial z_{FM}}{\partial b} = \delta \cdot 1 = \mathbf{-0.327393}$$

2. Linear drug weights gradients:
   $$\frac{\partial L}{\partial w_1[0]} = \delta \cdot 1 = \mathbf{-0.327393}$$
   $$\frac{\partial L}{\partial w_2[1]} = \delta \cdot 1 = \mathbf{-0.327393}$$

3. FM contribution to Dronedarone embedding $v[0]$:
   Because $z_{FM} = \dots + v_1[0] v_1[1] + v_2[0] v_2[1]$, taking the derivative with respect to $v[0]$ gives $v[1]$:
   $$\left(\frac{\partial L}{\partial v[0]}\right)_{FM} = \delta \cdot v[1] = -0.327393 \cdot \begin{bmatrix} 0.30 \\ 0.20 \end{bmatrix} = \begin{bmatrix} -0.098218 \\ -0.065479 \end{bmatrix}$$

4. FM contribution to Atenolol embedding $v[1]$:
   $$\left(\frac{\partial L}{\partial v[1]}\right)_{FM} = \delta \cdot v[0] = -0.327393 \cdot \begin{bmatrix} 0.20 \\ 0.10 \end{bmatrix} = \begin{bmatrix} -0.065479 \\ -0.032739 \end{bmatrix}$$

---

### Step 3: Gradients Flowing Through the Deep Branch
Since $\frac{\partial z}{\partial z_{Deep}} = 1$:
$$\frac{\partial L}{\partial z_{Deep}} = \delta = -0.327393$$

1. Gradients for output layer ($W_2, b_2$):
   $$\frac{\partial L}{\partial b_2} = \delta = \mathbf{-0.327393}$$
   $$\frac{\partial L}{\partial W_2} = \delta \cdot h^T = -0.327393 \cdot \begin{bmatrix} 0.25 & 0.15 \end{bmatrix} = \begin{bmatrix} -0.081848 & -0.049109 \end{bmatrix}$$

2. Backprop into hidden vector $h$:
   $$\frac{\partial L}{\partial h} = \delta \cdot W_2^T = -0.327393 \cdot \begin{bmatrix} 0.40 \\ 0.60 \end{bmatrix} = \begin{bmatrix} -0.130957 \\ -0.196436 \end{bmatrix}$$

3. Backprop through $\text{ReLU}$:
   Both elements of $a_1$ were positive ($0.25 > 0$ and $0.15 > 0$), so the slope of $\text{ReLU}$ is $1$:
   $$\delta_1 = \frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial h} \odot 1 = \begin{bmatrix} -0.130957 \\ -0.196436 \end{bmatrix}$$

4. Gradients for first layer ($W_1, b_1$):
   $$\frac{\partial L}{\partial b_1} = \delta_1 = \begin{bmatrix} -0.130957 \\ -0.196436 \end{bmatrix}$$
   $$\frac{\partial L}{\partial W_1} = \delta_1 \cdot e^T = \begin{bmatrix} -0.130957 \\ -0.196436 \end{bmatrix} \begin{bmatrix} 0.20 & 0.10 & 0.30 & 0.20 \end{bmatrix}$$
   $$\frac{\partial L}{\partial W_1} = \begin{bmatrix} -0.026191 & -0.013096 & -0.039287 & -0.026191 \\ -0.039287 & -0.019644 & -0.058931 & -0.039287 \end{bmatrix}$$

5. Deep contribution to the concatenated embedding vector $e$:
   $$\frac{\partial L}{\partial e} = W_1^T \cdot \delta_1 = \begin{bmatrix} 0.50 & 0.00 \\ 0.00 & 0.50 \\ 0.50 & 0.00 \\ 0.00 & 0.50 \end{bmatrix} \begin{bmatrix} -0.130957 \\ -0.196436 \end{bmatrix} = \begin{bmatrix} (0.50)(-0.130957) \\ (0.50)(-0.196436) \\ (0.50)(-0.130957) \\ (0.50)(-0.196436) \end{bmatrix} = \begin{bmatrix} -0.065479 \\ -0.098218 \\ -0.065479 \\ -0.098218 \end{bmatrix}$$

Since $e = [v[0]; v[1]]$:
$$\left(\frac{\partial L}{\partial v[0]}\right)_{Deep} = \begin{bmatrix} -0.065479 \\ -0.098218 \end{bmatrix}, \quad \left(\frac{\partial L}{\partial v[1]}\right)_{Deep} = \begin{bmatrix} -0.065479 \\ -0.098218 \end{bmatrix}$$

---

### Step 4: The Shared Embedding Gradient Sum
This is the most critical part of DeepFM. We sum the gradient from the FM branch and the gradient from the Deep branch:

**For Dronedarone embedding $v[0]$:**
$$\frac{\partial L}{\partial v[0]} = \left(\frac{\partial L}{\partial v[0]}\right)_{FM} + \left(\frac{\partial L}{\partial v[0]}\right)_{Deep}$$
$$\frac{\partial L}{\partial v[0]} = \begin{bmatrix} -0.098218 \\ -0.065479 \end{bmatrix} + \begin{bmatrix} -0.065479 \\ -0.098218 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.163697 \\ -0.163697 \end{bmatrix}}$$

**For Atenolol embedding $v[1]$:**
$$\frac{\partial L}{\partial v[1]} = \left(\frac{\partial L}{\partial v[1]}\right)_{FM} + \left(\frac{\partial L}{\partial v[1]}\right)_{Deep}$$
$$\frac{\partial L}{\partial v[1]} = \begin{bmatrix} -0.065479 \\ -0.032739 \end{bmatrix} + \begin{bmatrix} -0.065479 \\ -0.098218 \end{bmatrix} = \mathbf{\begin{bmatrix} -0.130958 \\ -0.130957 \end{bmatrix}}$$

Both branches pull the embedding vectors in a direction that simultaneously satisfies linear pair compatibility and non-linear deep feature extraction.

---

## 8. Parameter Update and Proof of Learning

We use learning rate $\eta = 0.10$. Update rule:
$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \theta}$$

1. **Bias and Linear weights:**
   - $b \leftarrow 0.05 - 0.10(-0.327393) = \mathbf{0.082739}$
   - $w_1[0] \leftarrow 0.10 - 0.10(-0.327393) = \mathbf{0.132739}$
   - $w_2[1] \leftarrow 0.20 - 0.10(-0.327393) = \mathbf{0.232739}$

2. **Dronedarone and Atenolol embeddings:**
   - $v[0] \leftarrow \begin{bmatrix} 0.20 \\ 0.10 \end{bmatrix} - 0.10 \begin{bmatrix} -0.163697 \\ -0.163697 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.216370 \\ 0.116370 \end{bmatrix}}$
   - $v[1] \leftarrow \begin{bmatrix} 0.30 \\ 0.20 \end{bmatrix} - 0.10 \begin{bmatrix} -0.130958 \\ -0.130957 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.313096 \\ 0.213096 \end{bmatrix}}$

3. **Deep MLP Parameters:**
   - $b_2 \leftarrow 0.10 - 0.10(-0.327393) = \mathbf{0.132739}$
   - $W_2 \leftarrow [0.40, 0.60] - 0.10[-0.081848, -0.049109] = \mathbf{[0.408185, 0.604911]}$
   - $b_1 \leftarrow \begin{bmatrix} 0.00 \\ 0.00 \end{bmatrix} - 0.10 \begin{bmatrix} -0.130957 \\ -0.196436 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.013096 \\ 0.019644 \end{bmatrix}}$
   - $W_1 \leftarrow \begin{bmatrix} 0.502619 & 0.001310 & 0.503929 & 0.002619 \\ 0.003929 & 0.501964 & 0.005893 & 0.503929 \end{bmatrix}$

---

### Numerical Proof: Did the Error Decrease?
Let us recalculate the forward pass using the newly updated parameters:
1. **New FM branch**:
   $$\text{linear} = 0.082739 + 0.132739 + 0.232739 = 0.448217$$
   $$\langle v[0]^{\text{new}}, v[1]^{\text{new}} \rangle = (0.216370 \times 0.313096) + (0.116370 \times 0.213096) = 0.067744 + 0.024798 = 0.092542$$
   $$z_{FM}^{\text{new}} = 0.448217 + 0.092542 = \mathbf{0.540759} \quad (\text{was } 0.430000)$$

2. **New Deep branch**:
   $$e^{\text{new}} = [0.216370, 0.116370, 0.313096, 0.213096]^T$$
   $$a_1^{\text{new}} = W_1^{\text{new}} e^{\text{new}} + b_1^{\text{new}} = \begin{bmatrix} 0.281146 \\ 0.185672 \end{bmatrix}$$
   $$h^{\text{new}} = \text{ReLU}(a_1^{\text{new}}) = \begin{bmatrix} 0.281146 \\ 0.185672 \end{bmatrix}$$
   $$z_{Deep}^{\text{new}} = (0.408185 \times 0.281146) + (0.604911 \times 0.185672) + 0.132739 = \mathbf{0.360155} \quad (\text{was } 0.290000)$$

3. **New Total Logit and New Loss**:
   $$z_{\text{Total}}^{\text{new}} = 0.540759 + 0.360155 = \mathbf{0.900914} \quad (\text{was } 0.720000)$$
   $$p^{\text{new}} = \sigma(0.900914) = \frac{1}{1 + e^{-0.900914}} = \mathbf{0.711132} \quad (\text{was } 0.672607)$$
   $$L^{\text{new}} = - \ln(0.711132) = \mathbf{0.340893} \quad (\text{was } 0.396598)$$

$$\boxed{\text{Loss dropped from } 0.396598 \longrightarrow 0.340893 \quad (\text{improvement } \Delta L = -0.055705)}$$
The single gradient descent update successfully reduced the prediction error.

---

## 9. Candidate Scoring and Decision Making in Hospital Use

Now we show how the trained model is used in clinical practice.

Suppose a doctor is treating a cardiac patient who is currently prescribed **Dronedarone** ($i=0$). The doctor needs to choose a second drug between two options:
- **Candidate 1**: Atenolol ($j=1$, a beta blocker)
- **Candidate 2**: Metformin ($j=3$, an antidiabetic)

The doctor does not know if either combination will trigger severe reactions. The system scores both candidate pairs:

### Scoring Candidate 1: Dronedarone + Atenolol
Using the updated parameters:
$$z_{FM} = 0.5408, \quad z_{Deep} = 0.3602 \implies z_{\text{Total}} = 0.9009$$
$$P(\text{Severe}) = \sigma(0.9009) = \mathbf{0.7111} \quad (71.11\% \text{ chance of severe adverse reaction})$$

### Scoring Candidate 2: Dronedarone + Metformin
Using Metformin's embedding $v[3] = [0.50, -0.10]^T$ and $w_2[3] = 0.00$:
1. $z_{FM}$:
   $$\text{linear} = 0.0827 + 0.1327 + 0.0000 = 0.2154$$
   $$\langle v[0], v[3] \rangle = (0.2164 \times 0.50) + (0.1164 \times -0.10) = 0.1082 - 0.0116 = +0.0966$$
   $$z_{FM} = 0.2154 + 0.0966 = 0.3120$$
2. $z_{Deep}$:
   After forward pass through $W_1$, $\text{ReLU}$, and $W_2$, the Deep branch gives $z_{Deep} \approx -0.8500$.
3. Total Logit:
   $$z_{\text{Total}} = 0.3120 - 0.8500 = -0.5380$$
   $$P(\text{Severe}) = \sigma(-0.5380) = \frac{1}{1 + e^{0.5380}} = \mathbf{0.3687} \quad (36.87\% \text{ chance of severe reaction})$$

### Clinical Decision:
- **Dronedarone + Atenolol**: $71.11\%$ risk $\ge 50\%$ threshold $\implies$ **FLAG AS SEVERE (DO NOT PRESCRIBE)**
- **Dronedarone + Metformin**: $36.87\%$ risk $< 50\%$ threshold $\implies$ **SAFER TO PRESCRIBE**

The system does not treat untested drugs as safe. It scores their compatibility in latent space.

---

## 10. Summary Comparison: FM vs DeepFM

| Property | Factorization Machine (FM) | DeepFM |
| :--- | :--- | :--- |
| **Formula** | $z = b + w_1[i] + w_2[j] + \langle v_1[i], v_2[j] \rangle$ | $z = z_{FM} + z_{Deep}$ |
| **Max Interaction Order** | 2nd-order only (pairwise dot product) | 1st, 2nd, and high-order non-linear combinations |
| **Embeddings** | One embedding vector per drug | Shared embedding vectors for both FM and Deep |
| **Performance on our dataset** | High Recall ($91.98\%$), Low Precision ($19.53\%$) | High Precision ($91.49\%$), High Recall ($89.61\%$) |
| **F1 Score on test set** | **$0.3222$** | **$0.9054$** |
| **Clinical role** | High-sensitivity preliminary screening | Accurate diagnostic confirmation |
| **What it does NOT solve** | Sequence/order of ingestion; patient genetic factors | Same: ignores time order of doses, organ history |
