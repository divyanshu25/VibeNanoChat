# Reinforcement Learning, from First Principles

*A primer on how nanochat does RL. Assumes you've understood pretraining and SFT and nothing past that.*

> **Note on file paths.** VibeNanoChat itself does not (yet) ship an RL stage — it covers pretraining and SFT-time evals. The RL code dissected below lives in the **upstream [nanochat](https://github.com/karpathy/nanochat) repo**, so every `path/to/file.py` reference here is relative to *that* repo, not this one.

---

## 1. Where we are, and why SFT isn't enough

By the time you reach RL you've done two things to the model:

1. **Pretraining** — predict the next token over a giant pile of internet text. The model learns language, facts, a sloppy world model.
2. **SFT (supervised fine-tuning)** — show the model a few thousand curated `(conversation, ideal_response)` pairs and again do next-token prediction, but now only on the assistant's tokens. The model learns the *format* of being a helpful assistant: turn-taking, tool calls, answering instead of continuing.

Both of these are the same hammer: **imitation**. You show the model a target string and say "make this string more likely." That's cross-entropy loss, the whole way down.

Here's the problem. Imitation has a ceiling, and the ceiling is the data. With SFT you can only ever teach the model to copy the answers you already wrote down. But for a task like grade-school math (GSM8K), what we actually want is not "memorize this one solution path." We want the model to **reason its way to the correct final number**, and there are a thousand valid reasoning paths to any given answer. We don't have demonstrations for all of them. More importantly, the thing we actually care about is a property of the *outcome* — "is the final number correct?" — not a property of any particular string.

SFT can't optimize that directly. SFT optimizes "match this string." We want to optimize "get the answer right, by whatever path." Those are different objectives, and the gap between them is exactly the gap that RL fills.

```
SFT:   here is the correct answer, make it more likely.   (imitation)
RL:    try to answer. I'll tell you if you were right.    (trial and error)
       now make the things that worked more likely.
```

That's the whole pitch. Let's build it up properly.

---

## 2. The setup: turning a chat model into an RL agent

We need to recast "answer a math question" as something RL can chew on. The vocabulary of RL is: **policy**, **action**, **trajectory**, **reward**.

- The **policy** is the model itself. Given a context, it outputs a probability distribution over the next token. Call it $\pi_\theta(\text{token} \mid \text{context})$ where $\theta$ are the model's weights. The policy *is* the network.
- An **action** is sampling one token from that distribution.
- A **trajectory** is a full rollout: start from the question, sample token after token until the model emits `<|assistant_end|>`. That sequence of tokens is one complete attempt at the problem.
- The **reward** is a single number we hand back *after* the trajectory finishes, scoring how good the attempt was.

For GSM8K the reward is dead simple. From `tasks/gsm8k.py`:

```python
def reward(self, conversation, assistant_response):
    is_correct = self.evaluate(conversation, assistant_response)
    return float(is_correct)   # 1.0 if final number matches, else 0.0
```

That's it. We parse the number after the `####` marker out of the model's completion, compare it to ground truth, and return `1.0` or `0.0`. Notice what we are **not** doing: we are not checking whether the reasoning matches, not scoring partial credit, not looking at style. Pure outcome. The model is free to reason however it likes; we only grade the final answer. This is the thing SFT structurally could not do.

> **A note on what makes a good reward.** GSM8K is a gift: the answer is a number, so it's *automatically verifiable*. We can check correctness with a regex, no human and no second model in the loop. This is why math and code are the poster children of LLM RL — the reward is cheap, objective, and un-gameable. The moment your reward needs a human or a learned reward model to judge it, everything gets harder and more expensive. nanochat sticks to the easy, verifiable case on purpose.

---

## 3. The core question: which tokens get credit?

Say we let the model attempt a problem and it produces a 200-token solution that arrives at the right answer. Reward = 1.0. 

Now what? We want to "make this work more likely." But the model didn't make one decision — it made 200 decisions, one per token. Which of those 200 tokens deserve credit for the success? The early ones that set up the approach? The arithmetic in the middle? The final number?

This is the **credit assignment problem**, and it's the central difficulty of RL. The reward arrives once, at the very end, but it has to be distributed back over every decision that led to it.

The beautiful, almost-too-simple answer that **policy gradients** give: *don't try to be clever about it. Give every token in the trajectory the same credit — the trajectory's total reward.* Scale each token's nudge by how well its rollout did: strongly upward for good rollouts, and weakly — or, once we add a refinement in §5, actively downward — for bad ones. Average over enough rollouts and the tokens that systematically lead to good outcomes float up, while the rest wash out as noise.

Let's make that precise.

---

## 4. Deriving the policy gradient (the one piece of math)

### 4.1 Step 0 — what a "gradient" means here

If you've done SFT you already know the training loop: compute a loss, call `.backward()`, the optimizer takes a step. The gradient ($\nabla_\theta$) is just a long vector — one number per weight in the model — that says "increase this weight to decrease the loss." Gradient *descent* follows that direction to reduce loss.

Here we want to *maximize* reward instead of minimize loss, so we follow the gradient in the *opposite* sign convention — **gradient ascent**. Mechanically it's identical: compute a gradient, step in it. The math below is about deriving *what that gradient is*.

---

### 4.2 Step 1 — the objective: maximize expected reward

When we say "maximize expected reward," concretely it means: over many different problems and rollouts, we want the average reward to be as high as possible. In math, if $\tau$ is one rollout (trajectory) and $R(\tau)$ is its reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau) \big]$$

$\mathbb{E}[\cdot]$ just means "average over many samples." The subscript $\tau \sim \pi_\theta$ says those samples are drawn from our current policy (the model with weights $\theta$).

We want $\nabla_\theta J(\theta)$ — the gradient of this average reward with respect to every weight.

---

### 4.3 Step 2 — a simpler idea first, and why we want more

You might think: out of 4 rollouts, if 2 got the answer right (reward=1) and 2 didn't (reward=0), why not just do SFT on the 2 correct ones? Ignore the failures, treat the successes as gold demonstrations, backpropagate normally.

This is a real algorithm — called **rejection sampling fine-tuning (RFT)** — and it works. In fact, with a pure 0/1 reward, REINFORCE (no baseline) is mathematically identical to it: a $R=0$ trajectory multiplies the gradient by zero, so it contributes nothing, and you're left reinforcing only the correct ones.

**But there's a better version.** The failures aren't just noise to be ignored — they're information. A rollout that got the wrong answer is actively telling you "don't do this." If you could also *push down* the probability of incorrect responses, you'd use the full signal from every rollout, not just half of it.

That calls for a framework where each rollout is weighted by an arbitrary *signed* number, not the binary keep-or-discard of RFT. The **log-derivative trick** (next) builds exactly that road: a gradient of the form *(weight) × (push this rollout up)*, where the weight can be any real number. A heads-up on what it does and doesn't buy us: with a plain 0/1 reward that weight is never negative, so the bare gradient *still* only pushes good rollouts up and ignores the rest — it does **not** suppress failures yet. The suppression arrives one section later, when a **baseline** (§5) makes the weight (then called the *advantage*) go negative for below-average rollouts. So the trick lays the road; the baseline is what finally drives down it.

---

### 4.4 Step 3 — the log-derivative trick (one algebraic identity)

Before the algebra, let's make one piece of notation concrete: **what is $p_\theta(\tau)$?**

It's the probability that your model (with weights $\theta$) generates that specific sequence of tokens. The model picks one token at a time, each with some probability. The probability of the whole trajectory is the product of every token's probability given everything before it:

$$p_\theta(\tau) = \pi_\theta(\text{"The"} \mid \text{question}) \times \pi_\theta(\text{"answer"} \mid \text{question, "The"}) \times \cdots \times \pi_\theta(\text{"10"} \mid \text{everything before})$$

Same idea as rolling two dice: probability of rolling 3 then 5 = $\frac{1}{6} \times \frac{1}{6}$. Multiply the individual probabilities.

This gives us **two distinct objects, and it's worth keeping them straight** (they look similar but mean different things):

- **$\pi_\theta(a_t \mid s_t)$ — the policy.** The per-token distribution the network outputs at one step: given the context $s_t$, the probability of each possible next token $a_t$. This is literally the model's softmax. It's *the* fundamental object — when we say "the policy," this is it.
- **$p_\theta(\tau)$ — the trajectory probability.** A single number for the *whole* rollout, obtained by multiplying the policy's per-step probabilities together: $p_\theta(\tau) = \prod_t \pi_\theta(a_t \mid s_t)$.

So $p_\theta$ is *built from* $\pi_\theta$ — they're related by that product, not the same thing. (In general RL the trajectory probability would also include environment-transition terms, but for an LLM the "environment" is deterministic — appending the chosen token simply *is* the next state — so the product is purely policy terms.)

The $\theta$ subscript on both is a reminder that these probabilities change when the weights change — that's what training does. A gradient step adjusts $\theta$ to make $p_\theta(\tau)$ higher for good trajectories and lower for bad ones.

---

There's a clever identity from calculus. For any function $p(\tau)$:

$$\nabla_\theta \, p(\tau) = p(\tau) \cdot \nabla_\theta \log p(\tau)$$

*Why is this true?* Chain rule. $\log p$ is a function of $p$, so $\frac{d}{d\theta} \log p = \frac{1}{p} \frac{dp}{d\theta}$, rearranged: $\frac{dp}{d\theta} = p \cdot \frac{d \log p}{d\theta}$. That's it. High school chain rule.

*Why is this useful?* First, a quick note on notation: $\mathbb{E}[f(\tau)]$ (expectation) and $\int p(\tau) f(\tau) \, d\tau$ (integral) are the same thing written two ways. Expectation *is* a weighted average, and in math a weighted average over a continuous space is written as an integral. So:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int p_\theta(\tau) \, R(\tau) \, d\tau$$

Same object, two notations. The integral form is useful here because it lets us write out the gradient explicitly. Now watch what happens:

$$\nabla_\theta J(\theta)
= \nabla_\theta \int p_\theta(\tau) \, R(\tau) \, d\tau
= \int \nabla_\theta p_\theta(\tau) \, R(\tau) \, d\tau$$

The second step moves the $\nabla_\theta$ inside the integral — valid because $R(\tau)$ doesn't depend on $\theta$ (the reward is fixed once the rollout is done), so it's just a constant being pulled outside.

Now apply the identity to swap $\nabla p$ for $p \cdot \nabla \log p$:

$$= \int p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \, d\tau$$

And now flip back to expectation notation — an integral weighted by $p_\theta(\tau)$ is exactly the definition of $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$. (The subscript $\tau \sim \pi_\theta$ is the conventional shorthand for "trajectories produced by rolling out the policy $\pi_\theta$" — which is sampling from $p_\theta$. People name it by the policy because the policy is the thing you actually control.)

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau)\, \nabla_\theta \log p_\theta(\tau) \big]}$$

The sampling is now *inside* the expectation and we only need to differentiate the $\log p_\theta(\tau)$ term — which is differentiable. The trick moved the differentiation off the sampling operation and onto the log-probability, which the model computes directly. (And in the next step we'll crack $\log p_\theta(\tau)$ open into the per-token policy terms.)

---

### 4.5 Step 4 — breaking the trajectory into tokens

The boxed result has a $\log p_\theta(\tau)$ — the log-probability of the *whole trajectory*. But the network doesn't output trajectory probabilities; it outputs per-token policy probabilities $\pi_\theta(a_t \mid s_t)$. So we crack the trajectory term open using the relationship from Step 3:

$$p_\theta(\tau) = \prod_{t} \pi_\theta(a_t \mid s_t)$$

where $a_t$ is the token sampled at step $t$ and $s_t$ is everything before it (the context). Taking the log of a product converts it to a sum (the standard $\log(ab) = \log a + \log b$):

$$\log p_\theta(\tau) = \sum_{t} \log \pi_\theta(a_t \mid s_t)$$

This is the key move: the trajectory log-prob is just the **sum of the per-token policy log-probs** — and those are exactly what the network produces. The gradient of a sum is the sum of the gradients:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

---

### 4.6 Putting it together: REINFORCE

Substituting back in:

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[ R(\tau) \cdot \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]$$

Read this one piece at a time:

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — the gradient that would make token $a_t$ *more likely* at step $t$. This is the direction to nudge the weights to assign higher probability to the token that was actually sampled.
- Multiply that direction by $R(\tau)$ — the reward for the whole trajectory.
- Sum over every token $t$ in the rollout.
- Average over many rollouts.

In plain English:

> **For every token in the rollout: compute the direction that makes that token more likely, then scale it by the trajectory's reward. If the rollout was good ($R=1$), push all its tokens up. If the reward was zero, do nothing. Repeat over many rollouts.**

That's it. That's REINFORCE. The gradient is computable because $\log \pi_\theta(a_t \mid s_t)$ is just the log-probability the model assigns to the token it sampled — something the model computes directly and can backpropagate through.

---

### 4.7 The connection to SFT that should click right now
For an autoregressive decoder (like the GPT here), the cross-entropy loss over one sequence is:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(a_t \mid a_{<t})$$

where $a_t$ is the true/target token at position $t$, and $a_{<t} = a_1, \dots, a_{t-1}$ is everything before it (the context). Usually you average instead of sum:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(a_t \mid a_{<t})$$


In SFT the training loss for one token is:

$$\mathcal{L}_\text{SFT} = -\log \pi_\theta(a_t \mid s_t)$$

Minimizing that is identical to maximizing $\log \pi_\theta(a_t \mid s_t)$, which is identical to the gradient direction above. So the per-token gradient in RL is **exactly the same computation as SFT**. The one difference:

```
SFT gradient:    gradient of log π  ×  1           (constant, always reinforce)
RL gradient:     gradient of log π  ×  R(τ)        (scale by reward — varies per rollout)
```

SFT is the special case where every demonstration gets reward = 1 (it's shown as correct by assumption). RL lets the reward vary: good rollouts get reinforced strongly, bad rollouts get suppressed, and the signal is proportional to outcome quality. Same engine, smarter throttle.

---

## 5. The variance problem, and the one trick that fixes it

REINFORCE works. But if you actually run it, it learns *slowly* and the training curve looks like a seismograph. The culprit is **variance**, and there's a one-line fix that is genuinely one of the prettiest tricks in RL. Let's see the problem first, because once you see it the fix is obvious.

Here's the thing that should bother you about the gradient we just derived. On GSM8K the reward is always 0 or 1 — **never negative**. So trace what the gradient actually does to each rollout:

- Correct rollout ($R=1$): push all its tokens **up**.
- Wrong rollout ($R=0$): multiply by zero → do **nothing**.

Stare at that for a second. **You never push anything down.** Every single step is some flavor of "make these sequences more likely," and the *only* thing separating a brilliant rollout from a mediocre one is how *hard* you shove it up. The model is supposed to deduce which behaviors are good purely from the fact that the good ones get shoved up slightly more often than the bad ones. That does work in the limit — the softmax is normalized, so relentlessly upvoting the good stuff implicitly downvotes everything else — but it's a maddeningly indirect signal. It's like coaching chess by only ever saying "nice!" at varying volumes and never once saying "no, not that."

Now make it concrete. Take a question the model already nails 90% of the time. Sample 16 rollouts → ~14 correct, ~2 wrong. The gradient cheerfully reinforces all 14 correct ones at full strength — even though the model *already knows how to solve this*. We're blowing almost the entire gradient budget applauding behavior that's already locked in, while the 2 genuinely informative failures contribute exactly nothing. Huge update, near-zero learning. **That's the variance problem: the size of the gradient has almost nothing to do with how much there actually is to learn.**

The fix is a **baseline**. Instead of scaling by the raw reward, scale by reward *minus some reference value* $b$:

$$\nabla_\theta J = \mathbb{E}\Big[ (R(\tau) - b) \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]$$

Two things make this beautiful.

**First, it's free — you can subtract any $b$ without biasing the gradient.** The intuition: REINFORCE only ever responds to *differences* between rollouts. Shift every reward down by the same constant and you haven't changed which rollouts are above the pack and which are below — you've only moved the goalposts equally for everyone. So the baseline can recalibrate *what counts as good*, but it cannot tug the gradient in any direction.

The formal reason is a tidy identity — **the expected score is zero**, $\mathbb{E}_\tau[\nabla_\theta \log p_\theta(\tau)] = 0$ — which makes the entire baseline term $\mathbb{E}[\,b \cdot \nabla_\theta \log p_\theta(\tau)\,]$ vanish. Here's why it's zero (the punchline is *not* that some expectation equals 1 — it's that a probability distribution always integrates to 1 *regardless of $\theta$*, so its gradient is 0):

$$
\mathbb{E}_{\tau}\big[\nabla_\theta \log p_\theta(\tau)\big]
= \int p_\theta \,\nabla_\theta \log p_\theta \, d\tau
= \int \nabla_\theta\, p_\theta \, d\tau
= \nabla_\theta \underbrace{\int p_\theta \, d\tau}_{=\,1\ \text{always}}
= \nabla_\theta\, 1 = 0
$$

The middle step runs the log-derivative trick backwards ($p\,\nabla\log p = \nabla p$); the last step is just "the gradient of a constant is zero."

**Second, it brings back the "down."** Pick $b$ to be roughly the average reward, and suddenly $R(\tau) - b$ is positive for better-than-average rollouts and *negative* for worse-than-average ones. Now the gradient pushes the good ones up **and the bad ones down** — both halves of the signal, finally. The quantity $R(\tau) - b$ gets a name: the **advantage**. It stops asking "was this rollout good?" and starts asking the only question that actually teaches: "was this rollout better or worse than what I'd normally do here?"

Now: what's a good baseline? The cleverest choice — and this is the central idea of **GRPO** (Group Relative Policy Optimization) — is to use *the average reward of other attempts at the same question*.

So the recipe becomes:

1. Take one question.
2. Sample a **group** of $G$ completions for it (nanochat uses `num_samples=16`).
3. Compute each completion's reward.
4. Use the **group's mean reward** as the baseline.
5. Advantage of completion $i$ = $R_i - \text{mean}(R)$.

This is gorgeous because the baseline is *per-question and self-calibrating*. An easy question where 15/16 rollouts succeed has a high baseline (~0.94), so the lone failure gets a large negative advantage ("you blew an easy one — fix that") while the successes get a tiny positive nudge ("nice, but everyone did that"). A hard question where only 1/16 succeeds has a low baseline (~0.06), so that one lucky correct rollout gets a big positive advantage ("you found the needle — do more of that"). The signal automatically focuses on the rollouts that are *surprising relative to their peers*. No value network, no extra model — just sample a bunch and subtract the mean.

In nanochat this is two lines, in `scripts/chat_rl.py`:

```python
mu = rewards.mean()
advantages = rewards - mu
```

That's the entire "GRPO." Let that sink in: the famous algorithm is `rewards - rewards.mean()`.

---

## 6. "GRPO," in quotes

We've actually already built the entire algorithm: REINFORCE + a group-mean baseline. But the code calls it "GRPO" (in scare quotes), and the literature has a zoo of acronyms — PPO, TRPO, GRPO, DAPO — that all sound like prerequisites. They're not. Each one is just **REINFORCE plus a patch for a specific pain**, and nanochat removes most of the patches because its setting never feels those pains. So before we say what got dropped, let's build the family tree from scratch. Then "GRPO in quotes" will read as "REINFORCE with the patches peeled back," which is exactly what it is.

### 6.1 The family tree: each algorithm just patches the last one's headache

The most useful thing to realize about the acronym soup — REINFORCE, TRPO, PPO, GRPO, DAPO — is that it isn't five rival algorithms you study separately. It's **one** algorithm, REINFORCE, with a chain of patches bolted on. Every node below is just "the previous one **plus** a fix for the specific headache it created." Here's the whole chain on one line; then we walk it node by node, and watch where nanochat hops off:

```
              reuse a batch         bound the           cheap             drop the             scaling
              (fix the bias)        drift              version             critic               fixes
  REINFORCE ───────────────────▶     TRPO     ───────▶  PPO        ──────▶  GRPO        ──────▶  DAPO
   (base)    via importance         (KL trust           (the                (group             (4 long-CoT
             sampling                region)            clip)               baseline)           tweaks)
```

Spoiler: nanochat rides only as far as GRPO's *one* idea and ignores everything else — but you can't see *why* it's safe to ignore the rest until you've met what it's ignoring. So let's meet them.

#### 6.1.1 REINFORCE — the base case

REINFORCE is the original policy gradient algorithm; everything else in this tree is a patch on top of it. The whole loop:

1. **Sample rollouts.** Let the current model attempt a batch of questions; each attempt is one trajectory.
2. **Score them.** Run the reward function on each — for GSM8K, 1.0 if the final number matches, else 0.0.
3. **Update.** For each token in each rollout, take the gradient of its log-probability (the direction that makes that token more likely) and scale it by the rollout's **reward**. Sum over all tokens and rollouts, step.

The core insight, which §4 derived: **it's the SFT cross-entropy gradient, scaled by a reward instead of treating every token as correct.** With a 0/1 reward that means a correct rollout (reward = 1) gets all its tokens pushed up, and a wrong rollout (reward = 0) contributes nothing — its gradient is multiplied by zero. Run enough questions through it and the distribution concentrates on the reasoning paths that work. You can run exactly this, today, on one GPU.

**The limitation that kicks off the chain.** REINFORCE is strictly **on-policy**: the gradient is only valid for the exact model that generated the rollouts. The instant you step, the model changes and the rollouts in your hand are stale — reusing them for a second step would bias the gradient. So one batch of rollouts buys exactly **one** step, then it's trash. Since generating rollouts is the expensive part (decoding tokens one at a time, across many samples), spending all that compute on a single update *stings* — and that sting is what the next node fixes.

#### 6.1.2 Importance sampling — fixes "a batch is only good for one step"

(Not an algorithm on its own — it's the patch the next three are built on, so it comes first.)

We'd love to take several gradient steps per batch, since rollouts are expensive to generate. The naive way — just keep stepping on the same data — is subtly wrong: after the first step the model has moved, so you're now training on samples drawn by an *older* version of itself. The gradient quietly points the wrong way, and nothing warns you.

The fix is one of the oldest tricks in statistics. If you have samples from an old distribution but want the average under a new one, reweight each sample by how much more (or less) likely the new model makes it:

$$\text{ratio} = \frac{\pi_\text{new}(a)}{\pi_\text{old}(a)} \quad\text{— "how much more likely is this token now than when I sampled it?"}$$

A token that's become 1.5× as likely since sampling counts 1.5×; one that's become half as likely counts 0.5×. Reweighting every token this way cancels the bias, and stale data becomes usable again. So REINFORCE's objective `reward · logp` simply becomes `reward · ratio`. That's the whole idea.

And it barely touches the gradient. Differentiating the ratio with the same `∇p = p·∇log p` identity from §4.4 (with $\pi_\text{old}$ held constant) gives:

$$\nabla\,\text{ratio} = \frac{\nabla\pi_\text{new}}{\pi_\text{old}} = \frac{\pi_\text{new}\cdot\nabla\log\pi_\text{new}}{\pi_\text{old}} = \text{ratio}\cdot\nabla\log\pi_\text{new}$$

so REINFORCE's per-token gradient `reward · ∇logp` becomes `reward · ratio · ∇logp` — the very same gradient, with the ratio bolted on as a reweighting factor.

In code it's a one-line change, and you compute the ratio in log-space for stability:

```python
logp   = -cross_entropy(logits, targets)   # log π_new — same forward pass as REINFORCE/SFT
ratio  = torch.exp(logp - logp_old)        # = π_new/π_old; logp_old logged at sampling, detached
pg_obj = (ratio * rewards).sum()           # was: (logp * rewards).sum()
```

(`exp(logp - logp_old)` is `π_new/π_old` written in log-space — subtract the logs, exponentiate once — because the model emits log-probs and this dodges the underflow of multiplying hundreds of tiny numbers.)

Now the payoff, and why nanochat skips this whole node. On the *first* step, before the model has moved, `π_new = π_old`, so every ratio is exactly 1 and `reward · ratio · ∇logp` collapses straight back to plain REINFORCE. The ratio only earns its keep on the *second* step onward — once you reuse the batch and the two policies have drifted apart. nanochat takes exactly one gradient step per batch and then resamples fresh, so its ratio is always 1, and its code uses the plain on-policy objective without ever computing one.

#### 6.1.3 TRPO — fixes "a reused-data step can leap too far"

Reusing a batch — the previous fix — opened a new wound. The moment the current policy drifts from the one that generated the data, the ratio stops being near 1: a token the old model thought unlikely but the new one now loves produces a *huge* ratio, the gradient's variance detonates, and one oversized step can wreck the model. So we need a cap — don't let the policy move *too far* in a single round of reuse.

The obvious cap, a small fixed learning rate, doesn't work. The same step size that's harmless early in training can, a few updates later, shove the next-token distribution off a cliff (the model starts repeating one token or spewing garbage) — because a fixed move in *weight space* can be a negligible change in behavior in one place and a catastrophic one in another. So we can't budget the step by how far the *weights* move; we have to budget it by how far the model's *behavior* moves. And to do that we need a way to measure "how different are two policies" — which is the **KL divergence**.

> **Quick aside — what is "KL"?** Everyone uses it to mean "how different are two probability distributions," and that's all it is. Say at one position the old policy thinks the next token is `{cat: 0.6, dog: 0.4}` and the new one thinks `{cat: 0.5, dog: 0.5}`. KL gives you a single number for how far apart those two opinions are — `0` if identical, growing as they diverge. The formula is just a weighted average of the log-ratio of the two probabilities:
>
> $$\text{KL}(p \,\|\, q) = \sum_{v \in V} p(v)\,\log\frac{p(v)}{q(v)}$$
>
> Read it as: for each possible token $v$, look at how many times more (or less) likely $p$ thinks it is versus $q$ — that's $\log\frac{p(v)}{q(v)}$ — and average those log-ratios, weighting by how often $p$ actually produces that token. If $p$ and $q$ agree everywhere, every ratio is 1, every log is 0, and the sum is 0.
>
> For an LLM the "distribution" is the next-token softmax, so you compute this sum **over the whole vocabulary at one position**, then **add it up across all positions** in the sequence to get the KL for the whole response. (Computing the full vocab sum for every token, for two models, is pricey — so in practice people estimate it from just the *sampled* token's probabilities, but the exact definition is the sum above.)

With KL as the ruler, **TRPO** (Trust Region Policy Optimization) states its rule in one line: **never let a single update change the policy by more than a fixed KL budget** — say $\text{KL} = 0.01$. The catch is the weight-vs-behavior gap from above: "how many weight-units equals 0.01 of KL" isn't constant — it depends on where you are in weight space right now. So TRPO has to *estimate* it locally before stepping.

Here's how. Think of the KL as a bowl sitting in weight space. Right at your current position, TRPO fits a local quadratic approximation to that bowl — how fast does the bowl curve upward as you step away? The curvature is captured by the **Fisher Information Matrix (F)**, which you can think of as "the KL bowl's second-derivative, averaged over your data." If you know F at the current point, you can compute the exact direction and distance that keeps you inside the $\text{KL} = 0.01$ boundary: the **natural gradient** step $F^{-1} g$, where $g$ is the ordinary gradient. This is the direction that makes the most reward progress per unit of KL spent — steepest ascent through the behavior lens, not the weight lens.

The expensive part: F is a matrix with as many rows and columns as there are parameters. For a large model that's astronomical. TRPO never explicitly forms or inverts it — instead it uses **conjugate gradient** (an iterative solver) to approximate $F^{-1} g$ without materializing F. That's still expensive: several passes over the data, each doing a matrix-vector product with F (computed as a Hessian-vector product via automatic differentiation). After all that work you have a candidate step. TRPO then **actually applies the step, re-runs the new policy on the batch, and measures the real KL** — and if it overshot 0.01, or the objective didn't improve, it **halves the step and repeats**. That step → measure → halve loop is what literally runs.

All of that — Fisher estimation, conjugate gradient, line search — is the "expensive second-order bit," and it's exactly why people reached for PPO's cheaper clip instead. PPO threw the whole thing away and got 90% of the benefit with a one-line clamp.

#### 6.1.4 PPO — fixes "TRPO's second-order math is too expensive"

**PPO** (Proximal Policy Optimization) is the cheap, wildly popular approximation that made TRPO a museum piece. It keeps §6.1.2's reusable ratio objective but makes two moves: weight each token by a proper **advantage** instead of the raw reward, and replace TRPO's curvature math with a dead-simple **clip**. Take them in turn.

**1. The advantage — and the critic that produces it.** §5 already told us the weight should be an *advantage* (reward minus a baseline), so it's signed: positive when a token beat the baseline, negative when it fell short. The only open question is *what baseline*. PPO **learns** one — a second network, the **critic** (or value network), that reads a partial state (the prompt plus the tokens generated so far) and predicts how much reward to expect from here on. The advantage is then "what you actually got − what the critic predicted." The naive version is just that one number per rollout: `advantage = final_reward − V(s_0)`. But a rollout is 200 tokens long — why give every token the *same* advantage number computed only from the final outcome? A token at position 150 had almost no influence on whether the final answer was right, but a token at position 5 that set up the whole reasoning chain really mattered. **GAE** (Generalized Advantage Estimation) fixes this: instead of one global `reward − V(s_0)`, it computes a *per-token* advantage by looking at how much better or worse things went *from that token's position onward* compared to what the critic predicted — the "temporal difference" at each step, blended together with a decay factor so early tokens don't get full credit for everything that happened later. In practice it's just a weighted sum of per-step prediction errors along the rollout. The gist remains "actual − predicted," but done locally at every token position rather than globally at the end. A well-fit critic gives low-variance advantages — but it's a whole extra network to train and hold alongside the policy, and that cost is exactly what GRPO will target next (§6.1.5), swapping the critic out for §5's group-mean baseline.

**2. The clip — a cheap stand-in for TRPO's trust region.** Rather than measure KL and solve for a step size, PPO just forbids any token's ratio from leaving $[1-\epsilon,\,1+\epsilon] = [0.8,\,1.2]$ (with $\epsilon \approx 0.2$):

$$\mathcal{L}^{\text{PPO}} = \min\!\left(\underbrace{\text{ratio} \cdot A}_{\text{term 1: unclipped}},\quad \underbrace{\text{clip}(\text{ratio},\ 1{-}\epsilon,\ 1{+}\epsilon) \cdot A}_{\text{term 2: ratio clamped to } [0.8,\,1.2]}\right), \quad \epsilon = 0.2$$

There are **two** terms inside the `min`, not one. Term 1 is the plain importance-sampling objective from §6.1.2 — no clamping, ratio can go anywhere. Term 2 is the same thing but with the ratio hard-clamped to $[0.8, 1.2]$. The `min` picks whichever of the two is smaller.

To read the `min`, consider four cases — ratio can be inside or outside `[0.8, 1.2]`, and A can be positive or negative:

- **Good update, gone far enough** (A>0, ratio>1.2 — token's prob rose past the boundary): `min(1.5·A, 1.2·A) = 1.2·A` — picks the clipped term, gradient zero, stop pushing. ✓
- **Bad update made a mistake** (A<0, ratio>1.2 — model *increased* a bad token's prob): `min(1.5·(−1), 1.2·(−1)) = min(−1.5, −1.2) = −1.5` — picks the *unclipped* term, gradient flows, corrects back down. ✓
- **Bad update, suppressed far enough** (A<0, ratio<0.8 — token's prob fell past the boundary): `min(0.6·(−1), 0.8·(−1)) = min(−0.6, −0.8) = −0.8` — picks the clipped term, gradient zero, stop suppressing. ✓
- **Good update made a mistake** (A>0, ratio<0.8 — model *decreased* a good token's prob): `min(0.6·A, 0.8·A) = 0.6·A` — picks the *unclipped* term, gradient flows, corrects back up. ✓

The pattern: **clipping only kicks in when the ratio moved in the same direction as the advantage (a beneficial update that's gone far enough). When the ratio moved in the wrong direction — a mistake — the `min` picks the unclipped term and lets the gradient fix it.** With just the clipped term, cases 2 and 4 would have zero gradient and the mistakes would go uncorrected.

Another way to see it: `min(ratio·A, clip·A)` is always ≤ `ratio·A`, making it a **pessimistic lower bound** on the unclipped objective — you never over-claim progress, but you never block corrections. Net effect: TRPO's trust region, first-order and free, with no KL computed and no line search.

**The payoff** is the data reuse we were after all along: with the clip as a guardrail you can safely run *several* gradient steps over the same batch of rollouts before throwing it out and resampling. Squeezing many safe steps from each expensive batch is PPO's whole value proposition.

One more thing usually bundled into PPO-for-LLMs, while we're here: the **KL-to-reference leash.** Bounding the step size stops *blow-ups*, but not a slower failure — over many steps a policy chasing a loose or learned reward will creep toward degenerate text that scores high and reads like nonsense ("reward hacking"). The fix is to keep a *frozen copy* of the original SFT model and add a penalty for drifting too far from *it*. Careful: this is a **different KL from TRPO's**, and people mix them up constantly — TRPO bounds the gap between *consecutive* policies (step to step), while this one anchors to a *fixed* reference for the whole run. One limits how fast you move; the other ties you to base camp. (This is the "spring" version of a KL bound, vs. TRPO's hard wall.)

#### 6.1.5 GRPO — fixes "PPO hauls a costly second network (the critic)"

Step back and look at what that critic costs. It's a second model roughly as large as the policy, which in LLM-land means it **roughly doubles your memory footprint and a big chunk of your compute** — you're forwarding *and* backpropagating two giant networks every step. It has to be trained, so it adds its own loss and hyperparameters. And it's a chicken-and-egg headache: early in training the critic's predictions are garbage, so your advantages are garbage, so the policy gets noisy updates — and a critic that mis-estimates can drag the whole run sideways. For a *verifiable* task this is especially silly: you can just *check* whether the answer was right, so why train an entire network to *guess* how good a state is?

**GRPO** (Group Relative Policy Optimization) answers exactly that and **deletes the critic outright.** It still needs a baseline — but recall *why* we wanted one (the variance section): just to know whether a rollout did better or worse than "typical for this question." GRPO gets that for free from rollouts it's already generating: sample a **group** of $G$ completions per question and use the group's **mean reward** as the baseline. No value network, no GAE, no second model. That's literally the `rewards - rewards.mean()` we built earlier — the "critic" replaced by an empirical average over samples you were taking anyway. In practice this **cuts memory and model FLOPs by roughly half** versus PPO and removes the critic-instability failure mode entirely. The trade: you now need *several* completions per question to estimate that mean, costing extra sampling — but sampling is cheap next to hauling a second network. GRPO keeps PPO's clip and usually the KL-to-reference leash; the *only* thing it removes is the critic.

#### 6.1.6 DAPO — fixes "scaling up long-reasoning runs"

**DAPO** (Decoupled Clip and Dynamic sAmpling Policy Optimization, from ByteDance) is GRPO plus four practical fixes, each aimed at a real failure people hit when scaling up long chain-of-thought RL:

1. **Clip-Higher** — use a *higher* upper clip bound than lower (asymmetric $[1-\epsilon_\text{low},\,1+\epsilon_\text{high}]$ instead of symmetric $[0.8, 1.2]$). The symmetric clip quietly throttles *exploration*: it caps how fast a low-probability-but-good token can grow, so the policy collapses toward a few confident tokens and entropy dies. Loosening just the upper bound lets promising rare tokens climb.
2. **Dynamic Sampling** — the **dead-group problem** we hit earlier, fixed head-on. Groups where *every* completion is right (or every one wrong) have zero advantage spread and contribute *no gradient*. DAPO **filters those out and keeps sampling** until the batch is full of groups with a mix of right and wrong, so every step's batch is all live gradient, no wasted compute.
3. **Token-Level Policy Gradient Loss** — normalize the loss over *total tokens* rather than per-sequence-then-averaged, so long responses aren't under-weighted. **This is the one nanochat borrows** (see token-level normalization below).
4. **Overlong Reward Shaping** — when a response runs past the length limit and gets truncated, don't hand it a hard penalty (which punishes the model for a cutoff it couldn't see coming and adds reward noise); shape or mask the reward for overlong samples so the signal stays clean.

nanochat takes only #3. The other three matter most for the large-scale, long-chain-of-thought runs nanochat isn't trying to be.

### 6.2 So what does nanochat actually keep?

Here's the punchline: nanochat is strictly on-policy with **one gradient step per batch** — it never reuses rollouts. And once you never reuse data, the *entire* reuse-driven half of the family tree has nothing to do: there's no old policy to correct for (no importance ratio), nothing to explode (no clip), and — for short runs on a clean verifiable reward — no time to game anything (no reference leash). So nanochat keeps only GRPO's actual idea, the group-mean baseline, and throws the rest back. Concretely, here's what got dropped relative to textbook GRPO/PPO, and *why each is safe to drop here*:

**1. No KL-to-reference penalty / no reference model (the reward-hacking leash).**
No frozen copy of the SFT model, no KL leash. For short GSM8K runs with a clean, verifiable 0/1 reward there's nothing to game and no time to drift into gibberish, so the regularizer isn't worth its memory and compute cost.

**2. No PPO ratio + clip (the data-reuse machinery).**
The clip only earns its keep when you *reuse* rollouts across multiple gradient steps. nanochat is strictly on-policy — one step per batch, then resample — so the new policy and the sampling policy are identical, the ratio is exactly 1, and the clip would never trigger. Carrying it would be dead weight. The cost of this choice is sample efficiency (every step needs fresh, expensive rollouts); the payoff is that a whole layer of machinery evaporates.

**3. Token-level normalization (DAPO style), not sequence-level.**
When we average the loss, we divide by the **total number of valid tokens across the batch**, not by sequence and then across sequences. The practical effect: a single token in a long rollout carries the same weight as a single token in a short one. Sequence-level averaging would secretly upweight tokens in short responses, biasing the model toward terseness. Token-level keeps it honest.

**4. Plain mean-subtraction, not z-score.**
Classic GRPO standardizes the advantage: $(R - \mu)/\sigma$. nanochat uses just $(R - \mu)$. Dividing by the group's standard deviation turns out to inject its own subtle bias (it inflates the gradient on questions where rollouts happen to agree with each other), so dropping it is arguably *more* correct, not just simpler.

What's left after all that subtraction is essentially **REINFORCE with a group-mean baseline and token-level loss normalization**. Which is the point: nanochat is teaching you that the fancy acronym is a thin wrapper around a very old, very simple idea.

---

## 7. The full loop, end to end

Before the code, here's the shape of the whole thing. RL is an **outer loop** that alternates two phases — **rollout** (generate experience) and **learn** (one gradient step on it) — and then immediately goes back for fresh experience:

```
start from the SFT model
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 1 — rollout (no gradients):                       │
│     pick a batch of questions                            │ ◄──┐
│     for each question, sample a GROUP of completions     │    │
│     reward each completion                               │    │
│     advantage = reward − (mean reward of ITS OWN group)  │    │
│                                                          │    │  on-policy:
│  PHASE 2 — learn (gradients on):                         │    │  data is
│     teacher-force the sampled tokens back through         │    │  re-sampled
│     the model, scale each token's cross-entropy by        │    │  every step
│     its advantage, accumulate, and take ONE optimizer step│    │
└──────────────────────────────┬───────────────────────────┘    │
                               └──────── updated model ──────────┘
```

The single most important thing to internalize before reading the code: **once Phase 1 finishes sampling, a rollout is just a frozen list of token IDs — and Phase 2 treats it exactly like an SFT example.** Same teacher forcing, same shift-by-1, same cross-entropy. The model is trained to predict *its own samples*; the only twist is that each token's loss is scaled by the rollout's advantage. There is no separate ground-truth target in RL — the model's own generation *is* the target. (More on this in Phase 2.)

Now let's walk the actual code path in `scripts/chat_rl.py`.

### 7.1 Phase 1 — Rollout: generate experience (`get_batch`, under `@torch.no_grad()`)

```
for each training question:
    1. render the prompt, but DROP the reference answer
    2. sample 16 completions at temperature 1.0
    3. score each completion -> reward in {0, 1}
    4. advantage = reward - mean(reward)
    5. yield (tokens, inputs, targets, rewards, advantages)
```

**Step 1 — prime the model for a completion.** We take the conversation and strip the assistant's gold answer off the end, keeping only the `<|assistant_start|>` token so the model is poised to write its own attempt. From `nanochat/tokenizer.py`:

```python
def render_for_completion(self, conversation):
    ...
    messages.pop()                       # remove the gold assistant message
    ids, mask = self.render_conversation(conversation)
    assistant_start = self.encode_special("<|assistant_start|>")
    ids.append(assistant_start)          # prime for completion
    return ids
```

This is the crucial difference from SFT. In SFT the gold answer is present and we compute loss against it. In RL **the gold answer is deleted** — the model has to produce its own, and we'll grade it. The training data is only ever used as a question bank and an answer key for the reward function; the model never imitates the reference solution.

**Step 2 — sample a group of rollouts.** We ask the inference `Engine` for 16 completions at `temperature=1.0`. Temperature matters: at temp 0 the model is greedy and all 16 rollouts would be near-identical — no diversity, no group to compare against, no learning signal. We *want* exploration here, so we sample hot. The engine (`nanochat/engine.py`) does a single prefill of the shared prompt, then clones that KV cache across the sample batch and decodes the rollouts in parallel — in practice in chunks of `device_batch_size` to avoid OOM (8 at a time here, looped `num_samples // device_batch_size` times to reach all 16). It also runs the **tool-use state machine** mid-rollout: when the model emits `<|python_start|>...<|python_end|>` the engine actually executes the calculator and *forces* the result tokens back into the stream.

**Step 3 — reward.** Decode each completion, run `task.reward(...)`, collect a `(16,)` vector of 0s and 1s.

**Step 4 — advantage.** `advantages = rewards - rewards.mean()`. The crucial detail: `get_batch` computes this **one question at a time**, so `rewards.mean()` is the mean over *that question's own group of completions* — never pooled across different questions. Each group is normalized against its own baseline. A consequence worth knowing: if every completion in a group gets the *same* reward (all correct, or all wrong), the mean equals every reward, so **all advantages are zero** and that question contributes *no gradient at all* this step. Small groups go unanimous more often, which is one reason the group size (`num_samples`) is a real knob — too small and a chunk of your questions silently go dead each step.

### 7.2 Phase 2 — Learn: the policy-gradient step

The sampled tokens from Phase 1 are fed back through the model — this time **with gradients on**, unlike the `no_grad` sampling pass — so that `.backward()` can compute $\nabla_\theta \log \pi_\theta$ for each sampled token. The objective is five lines:

```python
# logp of every sampled token  (same -cross_entropy as SFT, just per-token)
logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)   # (B, T)

# scale each token's logp by its rollout's advantage and sum
pg_obj = (logp * advantages.unsqueeze(-1)).sum()

# token-level normalization across the full step
# num_valid         — valid (non-masked) tokens in this sub-batch
# num_passes        — sub-batches one question's rollouts are split into
#                     (= num_samples // device_batch_size, e.g. 16//8 = 2)
# examples_per_rank — questions this GPU handles per step
#                     (= examples_per_step // num_gpus;
#                      e.g. 16 questions across 8 GPUs → 2 questions per GPU)
# product = total valid tokens across ALL questions and ALL sub-batches on this GPU,
# so gradient magnitude is invariant to how you chunk device_batch_size or GPU count.
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

loss = -pg_obj   # maximize objective → minimize its negation
loss.backward()
```

- **`loss_reduction='none'`** — cross-entropy returns a scalar by default; we need the per-token NLL so each token can carry its own advantage weight. The `view_as(inputs)` reshapes it back to `(B, T)`.
- **`logp * advantages.unsqueeze(-1)`** — broadcasts the per-rollout advantage across every token in that rollout. This is exactly $A_i \cdot \nabla\log\pi_\theta(a_t \mid s_t)$ from the derivation in §4.
- **Token-level normalization** — dividing by valid-token count (not sequence count) so a long rollout and a short one each contribute proportionally, not disproportionately. This is the DAPO technique from §6.2.
- **Negate** — PyTorch minimizes; we want to maximize the policy-gradient objective.

That's the whole gradient. No value head, no GAE, no clipping. Just SFT's cross-entropy, per-token, weighted by advantage.

### 7.3 The masking detail that's easy to miss

```python
targets = ids[:, 1:].clone()
targets[mask_ids[:, 1:] == 0] = -1   # -1 is the ignore_index
```

The engine returns a mask that is `0` for two kinds of tokens: (a) **prompt tokens**, and (b) **tool-use output tokens** that the engine *forced* into the stream (the calculator results). Setting their targets to `-1` makes `cross_entropy(ignore_index=-1)` skip them completely.

The reason is fundamental to policy gradients: $\log \pi_\theta(a_t \mid s_t)$ only makes sense for tokens the model **actually sampled**. The prompt was given, not chosen. The calculator outputs were injected by the engine, not generated by the model. Backpropagating through either would corrupt the gradient — you'd be crediting or blaming the model for tokens it never decided. One line, but load-bearing.

### 7.4 Accumulate, step, repeat

Two batch-size knobs that look similar but do different things:

- **`examples_per_step`** (default 16) — how many *questions* feed one `optimizer.step()`. This is the math knob: wider means each update sees a broader spread of problems.
- **`device_batch_size`** (default 8) — how many sequences fit in a single GPU forward pass. Pure memory bookkeeping; doesn't touch the math.

When `examples_per_step × num_samples` exceeds `device_batch_size`, the loop splits into sub-batches and accumulates gradients before the single `optimizer.step()`. Multiple forward/backward passes, one weight update. The chunking changes nothing about what gets optimized.

> **Concretely:** 2 questions × 4 samples = 8 sequences. If `device_batch_size=8`, it's one forward pass. If `device_batch_size=4`, it's two passes with gradient accumulation. Either way: one `optimizer.step()`, then throw the rollouts out and resample from the updated model.

Three practical notes:
- **Tiny LR.** `init_lr_frac=0.05` starts at 5% of the SFT rate, ramps to zero. RL gradients are noisy and the SFT model is already good — gentle nudges, not bulldozing.
- **No optimizer state saved.** The checkpoint skips Adam/Muon moments. On-policy runs are short; not worth the disk.
- **No fp16.** The gradient-scaler dance used in pretraining isn't wired up for RL. Use bf16 or fp32.

---

## 8. How do we know it's working? `pass@k`

Every `eval_every` steps we run the held-out test set and compute **pass@k**: out of $k$ sampled attempts at a problem, did *at least one* get it right? (See `run_gsm8k_eval`.)

```
pass@1  -> model is right on a single try
pass@8  -> at least one of 8 tries is right
```

This is the right lens for RL, and the relationship between the two numbers tells a story:

- **`pass@k` (large k) is your ceiling** — it measures whether the *capability* to solve the problem exists anywhere in the policy's distribution. Pretraining + SFT mostly set this.
- **`pass@1` is your floor** — it measures whether the model reliably *commits* to the right path on the first try.

The whole job of this flavor of RL is to **drag `pass@1` up toward `pass@k`**. The model already *can* solve these problems sometimes (that's why some rollouts get reward 1). RL doesn't teach a brand-new skill so much as it **sharpens the distribution onto the reasoning paths that already work**, making the good outcome the *likely* outcome instead of the occasional lucky one. You're not expanding what's possible; you're concentrating probability mass on the parts of the policy that succeed.

---

## 9. The one-paragraph version

You did SFT, which is imitation and therefore capped by your demonstrations. RL lets you optimize the *outcome* directly: let the model attempt a verifiable problem, score the attempt with a reward, and reuse your SFT cross-entropy gradient — but scaled by that reward — so successful rollouts get reinforced and failures suppressed. Sample a group of attempts per question and subtract the group's mean reward to get a low-variance **advantage** (that's the whole of "GRPO"). Strip away the trust region, the PPO clip, the z-score, and the value network — none are needed when you're on-policy with a clean reward — and you're left with REINFORCE plus a group baseline and careful token masking. Watch `pass@1` climb toward `pass@k` and you'll know it's concentrating probability onto the paths that already work.

```
pretraining   ->  learn language          (imitate the internet)
SFT           ->  learn to be an assistant (imitate good answers)
RL            ->  learn to be RIGHT        (try, get scored, reinforce what worked)
```

---

## 10. Where to look in the code

| Concern | File |
|---|---|
| The whole RL training loop | `scripts/chat_rl.py` |
| Reward function (verifiable 0/1) | `tasks/gsm8k.py` |
| Prompt priming (drop the gold answer) | `nanochat/tokenizer.py` → `render_for_completion` |
| Batched rollout sampling + tool-use machine | `nanochat/engine.py` → `generate` / `generate_batch` |
| Per-token loss (`loss_reduction='none'`) | `nanochat/gpt.py` → `forward` |

Run it:

```bash
# 1 GPU
python -m scripts.chat_rl

# 8 GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
```
